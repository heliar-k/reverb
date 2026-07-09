// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "reverb/cc/shm/shm_server.h"

#include <cstring>
#include <poll.h>
#include <sched.h>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {

namespace {

// Build the ShmColumn spec (proto) from a TensorBuffer.
::reverb::tensor::SignatureProto::TensorSpec TensorSpecFromBuffer(
    const TensorBuffer& buf) {
  ::reverb::tensor::SignatureProto::TensorSpec spec;
  spec.set_dtype(DataTypeToProto(buf.dtype()));
  auto* shape = spec.mutable_shape();
  for (int64_t d : buf.shape()) shape->add_dim(d);
  return spec;
}

}  // namespace

ShmServer::ShmServer(std::shared_ptr<Table> table, std::string socket_path,
                     ShmBytePool pool, ShmBootstrapServer bootstrap)
    : table_(std::move(table)),
      socket_path_(std::move(socket_path)),
      pool_(std::move(pool)),
      bootstrap_(std::move(bootstrap)) {}

// static
absl::StatusOr<std::unique_ptr<ShmServer>> ShmServer::Create(
    std::shared_ptr<Table> table, const std::string& socket_path) {
  if (table == nullptr) {
    return absl::InvalidArgumentError("table must not be null");
  }
  REVERB_ASSIGN_OR_RETURN(ShmBootstrapServer bootstrap,
                          ShmBootstrapServer::Create(socket_path));

  // The pool name is keyed by the server PID alone (A3).
  std::string pool_name =
      absl::StrCat("/reverb_shm_pool_", getpid());
  REVERB_ASSIGN_OR_RETURN(
      ShmBytePool pool, ShmBytePool::Create(pool_name));
  return absl::WrapUnique(
      new ShmServer(std::move(table), socket_path, std::move(pool),
                    std::move(bootstrap)));
}

ShmServer::~ShmServer() { Stop(); }

absl::Status ShmServer::Start() {
  if (running_.exchange(true)) {
    return absl::FailedPreconditionError("ShmServer already started");
  }
  dispatch_thread_ = std::thread([this] { DispatchLoop(); });
  return absl::OkStatus();
}

void ShmServer::Stop() {
  if (!running_.exchange(false)) return;
  if (dispatch_thread_.joinable()) dispatch_thread_.join();

  // Clean up clients: release any outstanding offsets and close fds. The rings
  // and pool are unlinked by their owners' destructors when `clients_` clears.
  for (auto& c : clients_) {
    if (!c->outstanding_offsets_.empty()) {
      pool_.ReleaseAll(
          std::vector<uint64_t>(c->outstanding_offsets_.begin(),
                                c->outstanding_offsets_.end()));
      c->outstanding_offsets_.clear();
    }
    if (c->fd >= 0) close(c->fd);
  }
  clients_.clear();
}

void ShmServer::DispatchLoop() {
  while (running_.load()) {
    TryAccept();
    for (size_t i = 0; i < clients_.size(); i++) {
      HandleClientRequests(i);
      FlushOutbox(*clients_[i]);
    }
    // ponytail: sched_yield keeps the loop hot without pegging a core. A
    // blocking poll on all ring fds would be cheaper CPU but needs eventfd
    // plumbing per ring; not worth it for v1 dispatch throughput.
    sched_yield();
  }
}

bool ShmServer::TryAccept() {
  // Poll the listen fd with zero timeout (non-blocking): only accept when a
  // client is already waiting, so the dispatch loop never stalls on accept.
  struct pollfd pfd;
  pfd.fd = bootstrap_.listen_fd();
  pfd.events = POLLIN;
  if (poll(&pfd, 1, 0) <= 0) return false;  // nothing waiting / error

  auto a = bootstrap_.Accept();
  if (!a.ok()) return false;
  auto [client_fd, client_pid] = std::move(a).value();

  auto hello = RecvHello(client_fd);
  if (!hello.ok()) {
    close(client_fd);
    return false;
  }
  auto version_ok = CheckProtocolVersion(hello->protocol_version());
  if (!version_ok.ok()) {
    close(client_fd);
    return false;
  }

  ShmSegmentNames names = MakeShmNames(getpid(), client_pid);
  // Create this client's two rings. (The pool was created once at server init;
  // its name is shared with all clients.)
  auto c2s = Ring::Create(names.c2s);
  if (!c2s.ok()) {
    close(client_fd);
    return false;
  }
  auto s2c = Ring::Create(names.s2c);
  if (!s2c.ok()) {
    close(client_fd);
    return false;
  }

  WelcomeResponse welcome;
  welcome.set_pool_shm_name(pool_.name());
  welcome.set_c2s_shm_name(names.c2s);
  welcome.set_s2c_shm_name(names.s2c);
  auto send = SendWelcome(client_fd, welcome);
  if (!send.ok()) {
    close(client_fd);
    return false;
  }

  auto state = std::make_unique<ClientState>();
  state->fd = client_fd;
  state->client_pid = client_pid;
  state->conn.c2s = std::move(*c2s);
  state->conn.s2c = std::move(*s2c);
  state->conn.pool_shm_name = pool_.name();
  // The server keeps its own owner pool (`pool_`); the client-side pool handle
  // in `conn` is left default (the server never reads sample bytes via it).
  clients_.push_back(std::move(state));
  return true;
}

void ShmServer::HandleClientRequests(size_t client_id) {
  ClientState& state = *clients_[client_id];
  // Drain everything currently readable (non-blocking). Ring::Read returns
  // NotFound("NOT_READY") when empty (spec §3.1) — that is the normal "no work"
  // signal, not an error.
  while (running_.load()) {
    MsgType type;
    std::string payload;
    absl::Status s = state.conn.c2s.Read(&type, &payload);
    if (!s.ok()) {
      // NOT_READY => ring empty, move on. Anything else is logged and we stop
      // draining this client this pass (a corrupted continuation would spam).
      if (!absl::IsNotFound(s)) {
        REVERB_LOG(REVERB_WARNING)
            << "ShmServer: C2S read error for client " << client_id << ": "
            << s;
      }
      return;
    }
    switch (type) {
      case SAMPLE: {
        ShmSampleRequest req;
        if (!req.ParseFromString(payload)) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: malformed ShmSampleRequest from client "
              << client_id;
          break;
        }
        auto st = HandleSample(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleSample failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case RELEASE: {
        ShmReleaseRequest req;
        if (!req.ParseFromString(payload)) break;
        auto st = HandleRelease(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleRelease failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case CLOSE:
        // ponytail: v1 just stops draining; full disconnect cleanup is ticket ⑥.
        return;
      default:
        // INSERT/ALLOCATE/etc. are ticket ④; ignore for now.
        break;
    }
  }
}

void ShmServer::FlushOutbox(ClientState& state) {
  absl::MutexLock lock(&state.outbox_mu);
  while (!state.outbox.empty()) {
    auto& [type, body] = state.outbox.front();
    absl::Status s =
        state.conn.s2c.Write(static_cast<MsgType>(type),
                             absl::MakeSpan(body));
    if (!s.ok()) {
      // Still full — leave the front in place and try again next pass (§8.7).
      // ponytail: a full outbox eventually backpressures via pool exhaustion;
      // v1 does not bound the outbox size. Upgrade: ring-buffer + drop policy.
      return;
    }
    state.outbox.erase(state.outbox.begin());
  }
}

absl::Status ShmServer::EnqueueS2C(ClientState& state, MsgType type,
                                   absl::string_view body) {
  // Write the response to the S→C ring. Ring::Write busy-yields (blocks) when
  // the ring is full; it only returns ResourceExhausted for a message larger
  // than the whole ring (never for "temporarily full").
  //
  // ponytail: v1 therefore blocks the dispatch thread on a slow client's full
  // S→C ring rather than fully implementing the spec §8.7 non-blocking+
  // outbox scheme (Ring has no try-write API). The outbox is still populated
  // on the oversized-message path for safety. Ceiling: one slow client stalls
  // all others until its S→C drains. Upgrade path: add Ring::TryWrite (compare
  // head-tail free space against slot demand) and route full rings through
  // FlushOutbox each pass, per spec §8.7. The client drains S→C in a tight
  // poll, so in practice the ring rarely fills.
  absl::Status s = state.conn.s2c.Write(type, absl::MakeSpan(body));
  if (s.ok()) return absl::OkStatus();
  if (!absl::IsResourceExhausted(s)) {
    return s;  // genuine error
  }
  // Oversized for the ring: stash in outbox (FlushOutbox will retry; in
  // practice this is a config error, not a flow-control path).
  {
    absl::MutexLock lock(&state.outbox_mu);
    state.outbox.emplace_back(static_cast<uint16_t>(type),
                              std::string(body));
  }
  return absl::OkStatus();
}

absl::Status ShmServer::HandleSample(ClientState& state,
                                     const ShmSampleRequest& req) {
  // 1. Sample from the real Table. This may block on the rate limiter up to
  //    `timeout_ms` (ponytail: blocks all clients while one waits; per-client
  //    dispatch thread later, spec §8.7).
  absl::Duration timeout = (req.timeout_ms() < 0)
                               ? absl::InfiniteDuration()
                               : absl::Milliseconds(req.timeout_ms());
  Table::SampledItem item;
  absl::Status sample_status = table_->Sample(&item, timeout);
  if (!sample_status.ok()) {
    // Map timeout (and rate-limiter timeout) to a ShmError the client surfaces.
    ShmError err;
    err.set_code(absl::IsDeadlineExceeded(sample_status)
                     ? ShmError::DEADLINE_EXCEEDED
                     : ShmError::INTERNAL);
    err.set_message(std::string(sample_status.message()));
    std::string body;
    err.SerializeToString(&body);
    return EnqueueS2C(state, ERROR, body);
  }

  // 2. Unpack each column on the dispatch thread (A1: single-threaded alloc).
  //    Mirror AsSample (sampler.cc): walk flat_trajectory().columns(), for each
  //    chunk slice call UnpackChunkColumnAndSlice. Then concatenate the slices
  //    of a column into one TensorBuffer (the trajectory view), applying the
  //    squeeze flag — exactly as Sample::AsTrajectory does.
  internal::flat_hash_map<uint64_t, std::shared_ptr<ChunkStore::Chunk>> chunks;
  for (auto& chunk : item.ref->chunks()) chunks[chunk->key()] = chunk;

  ShmSampleResponse resp;
  ShmSample* sample = resp.add_samples();

  // SampleInfo: key, priority, times_sampled, probability, table_size,
  // rate_limited — mirrors AsSample (sampler.cc).
  auto* info = sample->mutable_info();
  info->mutable_item()->set_key(item.ref->key());
  info->mutable_item()->set_priority(item.priority);
  info->mutable_item()->set_times_sampled(item.times_sampled);
  info->set_probability(item.probability);
  info->set_table_size(item.table_size);
  info->set_rate_limited(item.rate_limited);

  const auto& columns = item.ref->flat_trajectory().columns();
  for (int ci = 0; ci < columns.size(); ci++) {
    const auto& column = columns[ci];
    // Gather the column's unpacked slices.
    std::vector<TensorBuffer> slices;
    slices.reserve(column.chunk_slices_size());
    for (const auto& slice : column.chunk_slices()) {
      auto it = chunks.find(slice.chunk_key());
      if (it == chunks.end()) {
        return absl::InternalError(
            absl::StrCat("ShmServer: chunk ", slice.chunk_key(),
                         " not found when unpacking item ", item.ref->key()));
      }
      slices.emplace_back();
      REVERB_RETURN_IF_ERROR(internal::UnpackChunkColumnAndSlice(
          it->second->data(), slice, &slices.back()));
    }

    // Concatenate the column's slices into one batched trajectory tensor
    // (matches Sample::AsTrajectory: single-slice columns move, multi-slice
    // Concat along dim 0). The squeeze is applied CLIENT-side by
    // Sample::AsTrajectory (we send the batched tensor + the squeeze flag),
    // so the server and client mirror the local AsSample path exactly.
    TensorBuffer column_tensor;
    if (slices.size() == 1) {
      column_tensor = std::move(slices[0]);
    } else {
      auto concat = TensorBuffer::Concat(slices);
      REVERB_RETURN_IF_ERROR(concat.status());
      column_tensor = *std::move(concat);
    }

    bool squeeze = column.squeeze();

    // 3. memcpy the result bytes into the pool (C3: refcount starts at 1).
    absl::string_view bytes = column_tensor.bytes();
    REVERB_ASSIGN_OR_RETURN(uint64_t offset, pool_.Allocate(bytes.size()));
    pool_.Ref(offset);  // C3: refcount = 1 for the outstanding sample bytes
    std::memcpy(pool_.At(offset), bytes.data(), bytes.size());

    state.outstanding_offsets_.insert(offset);

    ShmColumn* col = sample->add_columns();
    col->set_shm_offset(offset);
    col->set_length(bytes.size());
    *col->mutable_spec() = TensorSpecFromBuffer(column_tensor);
    col->set_squeeze(squeeze);
  }

  std::string body;
  resp.SerializeToString(&body);
  return EnqueueS2C(state, SAMPLE_RESP, body);
}

absl::Status ShmServer::HandleRelease(ClientState& state,
                                      const ShmReleaseRequest& req) {
  for (uint64_t offset : req.offsets()) {
    if (pool_.Unref(offset)) {  // ->0
      pool_.Deallocate(offset);
    }
    state.outstanding_offsets_.erase(offset);
  }
  return absl::OkStatus();
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
