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
#include <unistd.h>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/hash_map.h"
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
      case INSERT: {
        ShmInsertRequest req;
        if (!req.ParseFromString(payload)) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: malformed ShmInsertRequest from client "
              << client_id;
          break;
        }
        auto st = HandleInsert(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleInsert failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case ALLOCATE: {
        ShmAllocateRequest req;
        if (!req.ParseFromString(payload)) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: malformed ShmAllocateRequest from client "
              << client_id;
          break;
        }
        auto st = HandleAllocate(state, req);
        if (!st.ok()) {
          REVERB_LOG(REVERB_WARNING)
              << "ShmServer: HandleAllocate failed for client " << client_id
              << ": " << st;
        }
        break;
      }
      case CLOSE:
        // ponytail: v1 just stops draining; full disconnect cleanup is ticket ⑥.
        return;
      default:
        // Unknown msg type: ignore (forward-compat).
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

absl::Status ShmServer::HandleAllocate(ClientState& state,
                                       const ShmAllocateRequest& req) {
  // C4: the server is the sole pool allocator. Grant the offset, track it as
  // outstanding (refcount=1) so a client crash (⑥) reclaims it, and reply with
  // ALLOCATE_RESP. The client memcpy's insert bytes here, then sends INSERT and
  // waits for InsertAck.offsets_to_release before reusing the region (C2).
  REVERB_ASSIGN_OR_RETURN(uint64_t offset, pool_.Allocate(req.num_bytes()));
  pool_.Ref(offset);  // outstanding against client crash
  state.outstanding_offsets_.insert(offset);
  ShmAllocateResponse resp;
  resp.set_shm_offset(offset);
  std::string body;
  resp.SerializeToString(&body);
  return EnqueueS2C(state, ALLOCATE_RESP, body);
}

absl::Status ShmServer::HandleInsert(ClientState& state,
                                      const ShmInsertRequest& req) {
  // 1. Deserialize each referenced ChunkData from the pool (C4: client wrote
  //    serialized bytes at the granted offset). ChunkData is self-describing,
  //    so ShmChunkRef.specs/sequence_range/delta_encoded are redundant metadata
  //    — the deserialized proto carries everything the ChunkStore needs.
  //    ponytail: skip populating the redundant ShmChunkRef metadata client-side.
  internal::flat_hash_map<uint64_t, std::shared_ptr<ChunkStore::Chunk>> chunks;
  for (const ShmChunkRef& ref : req.chunks()) {
    if (ref.total_length() == 0) {
      return absl::InvalidArgumentError(
          "ShmServer::HandleInsert: zero-length chunk");
    }
    ChunkData cd;
    if (!cd.ParseFromArray(pool_.At(ref.shm_offset()),
                           static_cast<int>(ref.total_length()))) {
      return absl::InternalError(absl::StrCat(
          "ShmServer::HandleInsert: failed to parse ChunkData at offset ",
          ref.shm_offset(), " (len ", ref.total_length(), ")"));
    }
    if (cd.chunk_key() != ref.chunk_key()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ShmServer::HandleInsert: chunk_key mismatch (ref ",
          ref.chunk_key(), " vs proto ", cd.chunk_key(), ")"));
    }
    // ChunkStore::Chunk owns a copy of the ChunkData proto. Multiple items
    // may reference the same chunk_key; dedup so storage is shared.
    chunks.try_emplace(cd.chunk_key(),
                       std::make_shared<ChunkStore::Chunk>(std::move(cd)));
  }

  // 2. For each PrioritizedItem, gather its referenced chunks and dispatch via
  //    InsertOrAssignAsync. The table name comes from item.table(). The
  //    completion callback fires on the table worker's callback-executor thread
  //    — NOT the dispatch thread — so it must not write the S→C ring directly
  //    (Ring is SPSC, single producer = the dispatch thread). Instead it stashes
  //    an InsertAck into the client's mutex-protected outbox; FlushOutbox
  //    (called each dispatch pass) drains it to S→C. This preserves the
  //    single-producer invariant while honoring C2 (client waits for ACK before
  //    reusing offsets).
  //
  //    ponytail: one ACK per insert request, aggregating all item keys and all
  //    chunk offsets. The client correlates by waiting for the next INSERT_ACK
  //    after sending INSERT (SPSC ordering guarantees it matches). A per-item
  //    ACK would let the client release offsets earlier, but v1's
  //    one-INSERT-per-item writer means there is exactly one item per ACK
  //    anyway. Ceiling: a batched-INSERT client would hold all offsets until the
  //    whole batch acks. Upgrade: per-item ACKs keyed by item key.
  std::vector<uint64_t> chunk_offsets;
  chunk_offsets.reserve(req.chunks_size());
  for (const ShmChunkRef& ref : req.chunks()) {
    chunk_offsets.push_back(ref.shm_offset());
  }

  InsertAck ack_template;
  for (uint64_t off : chunk_offsets) ack_template.add_offsets_to_release(off);

  // Collect the keys to ACK. We count outstanding items; when all complete,
  // enqueue the aggregate ACK. ponytail: atomic counter + outbox push under
  // outbox_mu; the dispatch thread is the sole reader/drainer.
  int num_items = req.items_size();
  if (num_items == 0) {
    // No items: still ack so the client can release the chunk offsets it
    // allocated (e.g. a chunks-only prefetch — v1 writer doesn't do this, but
    // be defensive).
    InsertAck ack = ack_template;
    std::string body;
    ack.SerializeToString(&body);
    return EnqueueS2C(state, INSERT_ACK, body);
  }

  auto remaining = std::make_shared<std::atomic<int>>(num_items);
  auto ack_keys = std::make_shared<std::vector<uint64_t>>();
  ClientState* state_ptr = &state;
  auto offsets = std::make_shared<std::vector<uint64_t>>(std::move(chunk_offsets));

  for (const PrioritizedItem& item_proto : req.items()) {
    const std::string& table_name = item_proto.table();
    if (table_name != table_->name()) {
      // ponytail: v1 server owns exactly ONE table (spec §3.4). A mismatch is
      // a client error; surface via the ACK's empty key set (the client treats
      // a missing key as failure). Multi-table server is a later ticket.
      REVERB_LOG(REVERB_WARNING) << "ShmServer::HandleInsert: table '"
                                 << table_name << "' != server table '"
                                 << table_->name() << "'; skipping item "
                                 << item_proto.key();
      // Count as completed so the ACK still fires.
      if (remaining->fetch_sub(1) == 1) {
        InsertAck ack;
        for (uint64_t k : *ack_keys) ack.add_keys(k);
        for (uint64_t off : *offsets) ack.add_offsets_to_release(off);
        std::string body;
        ack.SerializeToString(&body);
        REVERB_RETURN_IF_ERROR(EnqueueS2C(*state_ptr, INSERT_ACK, body));
      }
      continue;
    }

    // Gather the chunks referenced by this item's trajectory.
    std::vector<std::shared_ptr<ChunkStore::Chunk>> item_chunks;
    std::vector<uint64_t> keys = internal::GetChunkKeys(item_proto.flat_trajectory());
    item_chunks.reserve(keys.size());
    for (uint64_t ck : keys) {
      auto it = chunks.find(ck);
      if (it == chunks.end()) {
        return absl::InternalError(absl::StrCat(
            "ShmServer::HandleInsert: item ", item_proto.key(),
            " references unknown chunk ", ck));
      }
      item_chunks.push_back(it->second);
    }

    TableItem table_item(item_proto, std::move(item_chunks));

    // The callback fires on the table callback-executor thread. It must not
    // touch the S→C ring directly; it pushes the completed key, and when the
    // last item completes it enqueues the aggregate ACK into the outbox.
    auto cb = std::make_shared<Table::InsertCallback>(
        [remaining, ack_keys, offsets, state_ptr](uint64_t completed_key) {
          ack_keys->push_back(completed_key);
          if (remaining->fetch_sub(1) == 1) {
            InsertAck ack;
            for (uint64_t k : *ack_keys) ack.add_keys(k);
            for (uint64_t off : *offsets) ack.add_offsets_to_release(off);
            std::string body;
            ack.SerializeToString(&body);
            // EnqueueS2C writes to the ring directly (dispatch thread) or
            // stashes in outbox on ResourceExhausted. Called off the dispatch
            // thread, the direct write would race the dispatch thread's S→C
            // producer. Route through the outbox unconditionally instead.
            absl::MutexLock lock(&state_ptr->outbox_mu);
            state_ptr->outbox.emplace_back(
                static_cast<uint16_t>(INSERT_ACK), std::move(body));
            // All inserts confirmed: drop the keepalive so the callbacks (and
            // what they capture) are reclaimed. This breaks the would-be
            // cycle (cb -> lambda -> ... ; the lambda does NOT capture cb).
            state_ptr->pending_insert_callbacks.clear();
          }
        });
    // Keepalive: InsertOrAssignAsync stores a weak_ptr; the table worker fires
    // the callback AFTER HandleInsert returns, so the shared_ptr must outlive
    // this function. Stash it on the client; cleared by the last callback.
    {
      absl::MutexLock lock(&state.outbox_mu);
      state.pending_insert_callbacks.push_back(cb);
    }

    bool can_insert_more = false;
    absl::Status s = table_->InsertOrAssignAsync(std::move(table_item),
                                                 &can_insert_more, cb);
    if (!s.ok()) {
      return s;
    }
    // ponytail: v1 ignores can_insert_more on the server side — the dispatch
    // thread reads one INSERT at a time and the table's pending_inserts_ queue
    // absorbs bursts (max_enqueued_inserts). Backpressure is enforced
    // client-side via the ACK wait. If the table queue fills, the client's
    // outstanding items pile up in in_flight_items_ and the writer's
    // local_can_insert_more_ gate kicks in. Upgrade: honor can_insert_more by
    // deferring the read of the next INSERT until a completion fires.
  }

  return absl::OkStatus();
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
