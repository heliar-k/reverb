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

#include "reverb/cc/shm/shm_client.h"

#include <cstring>
#include <memory>
#include <sched.h>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/errors.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.pb.h"  // ticket ⑩: MutatePrioritiesRequest/ResetRequest
#include "reverb/cc/sampler.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/trajectory_writer.h"
#include "reverb/cc/writer.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {

namespace {

// ponytail: poll a non-blocking Ring::Read with sched_yield until a message is
// ready (spec §3.1 / R5: the blocking policy is the caller's job, not Ring's).
// Mirrors the helper used in echo_test / byte_pool_echo_test.
//
// ticket ⑥ (spec §8.8): if `control_fd` >= 0, ALSO probe it each pass for
// server death. The liveness udsocket fd is kept open for the connection
// lifetime; when the server dies/closes it becomes EOF/HUP. Without this the
// poll below would spin forever on a dead server, leaving in-flight sample /
// insert requests hanging. On server death return UnavailableError so the
// caller surfaces a Python-raisable status instead of hanging.
//
// ticket ⑩ 死锁修复（方向 C）：`timeout` 是健壮性兜底上限，独立于请求自身的
// rate-limiter timeout。即便服务端 dispatch 被其他请求阻塞（或未来出现新的阻塞
// 路径），client 也不会无限忙等——超时返 DeadlineExceededError，让 Python 信号
// 处理能跑、`timeout` 命令能杀。调用者应传有限值（请求 timeout 与一个硬上限取
// 大者）；默认 InfiniteDuration 仅保留给无超时语义的老调用点。
constexpr absl::Duration kReadBlockingHardCap = absl::Seconds(60);

absl::Status ReadBlocking(Ring* ring, MsgType* msg_type, std::string* payload,
                          int control_fd = -1,
                          absl::Duration timeout = absl::InfiniteDuration()) {
  absl::Time deadline = absl::Now() + timeout;
  while (true) {
    absl::Status s = ring->Read(msg_type, payload);
    if (s.ok()) return absl::OkStatus();
    if (!absl::IsNotFound(s)) return s;  // real error
    if (control_fd >= 0 && IsPeerClosed(control_fd)) {
      return absl::UnavailableError("SHM server closed connection");
    }
    if (absl::Now() >= deadline) {
      return absl::DeadlineExceededError("ShmClient: response timed out");
    }
    sched_yield();
  }
}

}  // namespace

// ---- ShmSampler ----

ShmSampler::ShmSampler(ShmConnection* conn, std::string table_name,
                       int64_t max_samples, absl::Duration rate_limiter_timeout)
    : conn_(conn),
      table_name_(std::move(table_name)),
      max_samples_(max_samples),
      rate_limiter_timeout_(rate_limiter_timeout),
      samples_(/*capacity=*/8) {}

// static
absl::StatusOr<std::unique_ptr<ShmSampler>> ShmSampler::Create(
    ShmConnection* conn, const std::string& table_name,
    const Sampler::Options& options) {
  if (conn == nullptr) {
    return absl::InvalidArgumentError("conn must not be null");
  }
  int64_t max_samples = options.max_samples == Sampler::kUnlimitedMaxSamples
                            ? INT64_MAX
                            : options.max_samples;
  if (max_samples < 1) {
    return absl::InvalidArgumentError("max_samples must be >= 1");
  }
  auto s = absl::WrapUnique(
      new ShmSampler(conn, table_name, max_samples, options.rate_limiter_timeout));
  s->worker_thread_ = internal::StartThread("ShmSamplerWorker",
                                            [self = s.get()] { self->RunWorker(); });
  return s;
}

ShmSampler::~ShmSampler() { Close(); }

void ShmSampler::Close() {
  if (closed_.exchange(true)) return;
  samples_.Close();
  if (worker_thread_) worker_thread_.reset();  // joins
}

absl::Status ShmSampler::GetNextTrajectory(
    std::vector<TensorBuffer>* data,
    std::shared_ptr<const SampleInfo>* info) {
  std::unique_ptr<Sample> sample;
  if (!samples_.Pop(&sample)) {
    // Queue closed: either max_samples hit, cancelled, or a worker error.
    absl::ReaderMutexLock lock(mu_);
    if (returned_ == max_samples_) {
      return absl::OutOfRangeError("`max_samples` already returned.");
    }
    if (closed_.load()) {
      return absl::CancelledError("ShmSampler has been cancelled.");
    }
    return worker_status_.ok() ? absl::CancelledError("ShmSampler closed.")
                               : worker_status_;
  }
  REVERB_RETURN_IF_ERROR(sample->AsTrajectory(data));
  if (info != nullptr) *info = sample->info();

  absl::WriterMutexLock lock(mu_);
  ++returned_;
  if (returned_ == max_samples_) samples_.Close();
  return absl::OkStatus();
}

void ShmSampler::RunWorker() {
  // ponytail: reserve one slot per sample, mirroring LocalSamplerWorker. On a
  // server-side timeout with nobody waiting to pop, the reservation is kept
  // and reused by the retry (PushBatch consumes it; a retry that succeeds
  // pushes into the same reserved slot). A timeout with a waiter, or any other
  // error, is surfaced via worker_status_ and the queue is closed.
  while (true) {
    {
      absl::WriterMutexLock lock(mu_);
      // Stop fetching when we've already requested max_samples (mirrors
      // Sampler::RunWorker's progress_trigger: requested_ < max_samples_).
      if (closed_.load() || requested_ >= max_samples_ ||
          !worker_status_.ok()) {
        return;
      }
      if (!samples_.Reserve(1)) {
        return;  // queue closed
      }
      ++requested_;
    }
    auto result = FetchOne();
    if (!result.ok()) {
      absl::Status st = result.status();
      if (absl::IsDeadlineExceeded(st) &&
          samples_.num_waiting_to_pop() < 1) {
        // Nobody waiting: keep the reservation, rewind requested_, retry.
        absl::WriterMutexLock lock(mu_);
        --requested_;
        continue;
      }
      // Real error: record it, close the queue so GetNextTrajectory unblocks.
      // The outstanding reservation is dropped (queue is closing anyway).
      absl::WriterMutexLock lock(mu_);
      if (worker_status_.ok() && !absl::IsCancelled(st)) {
        worker_status_ = st;
      }
      samples_.Close();
      return;
    }
    std::vector<std::unique_ptr<Sample>> batch;
    batch.push_back(std::move(*result));
    samples_.PushBatch(&batch);
  }
}

absl::StatusOr<std::unique_ptr<Sample>> ShmSampler::FetchOne() {
  // 1. Send SAMPLE request on the sample C→S ring (blocks if the ring is
  //    full — natural backpressure, §8.7). Decision D: the sample flow has
  //    its OWN ring pair, so this write never races the insert worker's
  //    writes on a shared c2s ring.
  ShmSampleRequest req;
  req.set_table(table_name_);
  req.set_num_samples(1);
  req.set_timeout_ms(
      NonnegativeDurationToInt64Millis(rate_limiter_timeout_));
  std::string req_body;
  req.SerializeToString(&req_body);
  REVERB_RETURN_IF_ERROR(
      conn_->sample_c2s.Write(SAMPLE, absl::MakeSpan(req_body)));

  // 2. Poll the sample S→C ring for the response. A server-side timeout comes
  //    back as ERROR with DEADLINE_EXCEEDED; surface it so the worker/sampler
  //    maps it. ticket ⑩ 方向 C：传有限超时作健壮性兜底——rate_limiter_timeout_
  //    无限时（默认）用 kReadBlockingHardCap，避免服务端 dispatch 被卡住时
  //    client 无限忙等（方向 A 修了根因，但任何未来阻塞仍应有上限）。
  MsgType resp_type;
  std::string resp_body;
  absl::Duration smp_timeout = (rate_limiter_timeout_ == absl::InfiniteDuration())
                                   ? kReadBlockingHardCap
                                   : rate_limiter_timeout_ + kReadBlockingHardCap;
  REVERB_RETURN_IF_ERROR(
      ReadBlocking(&conn_->sample_s2c, &resp_type, &resp_body, conn_->control_fd,
                   smp_timeout));

  if (resp_type == ERROR) {
    ShmError err;
    if (!err.ParseFromString(resp_body)) {
      return absl::InternalError("ShmSampler: malformed ShmError");
    }
    if (err.code() == ShmError::DEADLINE_EXCEEDED) {
      return errors::RateLimiterTimeout();
    }
    // ticket ⑨: map routing/argument errors so unknown-table samples surface as
    // real Python exceptions (NotFound -> FileNotFoundError, InvalidArgument ->
    // ValueError) instead of a generic InternalError.
    if (err.code() == ShmError::NOT_FOUND) {
      return absl::NotFoundError(err.message());
    }
    if (err.code() == ShmError::INVALID_ARGUMENT) {
      return absl::InvalidArgumentError(err.message());
    }
    return absl::InternalError(
        absl::StrCat("ShmSampler: server error: ", err.message()));
  }
  if (resp_type != SAMPLE_RESP) {
    return absl::InternalError(
        absl::StrCat("ShmSampler: unexpected response type ", resp_type));
  }

  ShmSampleResponse resp;
  if (!resp.ParseFromString(resp_body) || resp.samples_size() < 1) {
    return absl::InternalError("ShmSampler: malformed ShmSampleResponse");
  }
  const ShmSample& shm_sample = resp.samples(0);

  // 3. Build TensorBuffers from the pool bytes (worker thread, no GIL). Each
  //    column's bytes are copied out of SHM BEFORE releasing, so the sample is
  //    self-owned once queued (matches the spec note: assemble then RELEASE).
  std::vector<std::vector<TensorBuffer>> column_chunks;
  column_chunks.reserve(shm_sample.columns_size());
  std::vector<uint64_t> offsets;
  offsets.reserve(shm_sample.columns_size());
  std::vector<bool> squeeze_columns;
  squeeze_columns.reserve(shm_sample.columns_size());

  for (const ShmColumn& col : shm_sample.columns()) {
    offsets.push_back(col.shm_offset());
    squeeze_columns.push_back(col.squeeze());

    // Reconstruct the batched TensorBuffer from spec + the SHM bytes at the
    // offset. The server sent the batched (un-squeezed) tensor; Sample's
    // AsTrajectory applies the squeeze using squeeze_columns, exactly as the
    // local AsSample path does.
    TensorSpec spec;
    REVERB_ASSIGN_OR_RETURN(spec.dtype, DataTypeFromProto(col.spec().dtype()));
    spec.shape.reserve(col.spec().shape().dim_size());
    for (int64_t d : col.spec().shape().dim()) spec.shape.push_back(d);
    std::string bytes(col.length(), '\0');
    std::memcpy(&bytes[0], conn_->pool.At(col.shm_offset()), col.length());
    // Each column arrives as a single pre-concatenated batched tensor; wrap it
    // as a one-chunk column so Sample::AsTrajectory returns it directly (and
    // applies squeeze when set).
    std::vector<TensorBuffer> chunks;
    chunks.emplace_back(spec, std::move(bytes));
    column_chunks.push_back(std::move(chunks));
  }

  // 4. Assemble the Sample (mirrors AsSample: column_chunks + squeeze_columns).
  auto info = std::make_shared<SampleInfo>(shm_sample.info());
  auto sample = std::make_unique<Sample>(std::move(info),
                                         std::move(column_chunks),
                                         std::move(squeeze_columns));

  // 5. RELEASE the pool offsets now that bytes are copied out (C3), on the
  //    sample C→S ring (decision D).
  ShmReleaseRequest rel;
  for (uint64_t off : offsets) rel.add_offsets(off);
  std::string rel_body;
  rel.SerializeToString(&rel_body);
  REVERB_RETURN_IF_ERROR(
      conn_->sample_c2s.Write(RELEASE, absl::MakeSpan(rel_body)));

  return sample;
}

// ---- ShmClient ----

ShmClient::~ShmClient() = default;

ShmClient::ShmClient(ShmConnection conn) : conn_(std::move(conn)) {}

// static
absl::StatusOr<std::unique_ptr<ShmClient>> ShmClient::Connect(
    const std::string& socket_path) {
  // Bootstrap handshake (retry briefly while the server binds the socket).
  // ticket ⑥: keep the udsocket fd open as the liveness signal the server
  // poll()s for crash/close detection (spec §8.8).
  // Retry briefly while the server binds the udsocket (up to 10s). Without a
  // wall-clock deadline a server that never starts would busy-wait forever.
  absl::Time deadline = absl::Now() + absl::Seconds(10);
  absl::StatusOr<ClientBootstrapResult> w;
  while (absl::Now() < deadline) {
    w = ClientBootstrapWithFd(socket_path, /*client_pid=*/getpid());
    if (w.ok()) break;
    sched_yield();
  }
  REVERB_RETURN_IF_ERROR(w.status());

  // Open the four per-flow rings (decision D: insert + sample each get their
  // own SPSC pair) + the pool (RW, C4). RAII guard closes the bootstrap fd if
  // any Open fails — otherwise the moved-from `w->fd` would leak (its struct
  // dtor does not close fds).
  struct FdGuard {
    int fd = -1;
    ~FdGuard() { if (fd >= 0) close(fd); }
  } fd_guard{w->fd};
  REVERB_ASSIGN_OR_RETURN(ShmBytePool pool,
                          ShmBytePool::Open(w->welcome.pool_shm_name()));
  REVERB_ASSIGN_OR_RETURN(
      Ring insert_c2s, Ring::Open(w->welcome.insert_c2s_shm_name()));
  REVERB_ASSIGN_OR_RETURN(
      Ring insert_s2c, Ring::Open(w->welcome.insert_s2c_shm_name()));
  REVERB_ASSIGN_OR_RETURN(
      Ring sample_c2s, Ring::Open(w->welcome.sample_c2s_shm_name()));
  REVERB_ASSIGN_OR_RETURN(
      Ring sample_s2c, Ring::Open(w->welcome.sample_s2c_shm_name()));

  ShmConnection conn;
  conn.insert_c2s = std::move(insert_c2s);
  conn.insert_s2c = std::move(insert_s2c);
  conn.sample_c2s = std::move(sample_c2s);
  conn.sample_s2c = std::move(sample_s2c);
  conn.pool = std::move(pool);
  conn.pool_shm_name = w->welcome.pool_shm_name();
  conn.control_fd = fd_guard.fd;  // ~ShmConnection closes it
  fd_guard.fd = -1;               // conn owns it now

  return absl::WrapUnique(new ShmClient(std::move(conn)));
}

absl::Status ShmClient::ServerInfo(std::vector<TableInfo>* table_info) {
  // ticket ⑧ step 2: on-demand SERVER_INFO round-trip (replaces the step-1
  // bootstrap snapshot). Rides the INSERT flow under insert_flow_mu like
  // ⑩/⑪'s control-plane ops — see MutatePriorities for the mutex rationale.
  // ServerInfoRequest is empty; ServerInfoResponse carries repeated TableInfo.
  // Always succeeds server-side (HandleServerInfo has no error path).
  ServerInfoRequest req;
  std::string body;
  req.SerializeToString(&body);

  absl::MutexLock lock(&conn_.insert_flow_mu);
  REVERB_RETURN_IF_ERROR(
      conn_.insert_c2s.Write(SERVER_INFO, absl::MakeSpan(body)));

  MsgType resp_type;
  std::string resp_body;
  // Same kReadBlockingHardCap cap as ⑩/⑪'s control-plane ACKs.
  REVERB_RETURN_IF_ERROR(
      ReadBlocking(&conn_.insert_s2c, &resp_type, &resp_body, conn_.control_fd,
                   kReadBlockingHardCap));

  if (resp_type == ERROR) {
    // HandleServerInfo has no error path, but defend against future ones.
    ShmError err;
    if (!err.ParseFromString(resp_body)) {
      return absl::InternalError(
          "ShmClient::ServerInfo: malformed ShmError");
    }
    return absl::InternalError(absl::StrCat(
        "ShmClient::ServerInfo: server error: ", err.message()));
  }
  if (resp_type != SERVER_INFO_RESP) {
    return absl::InternalError(absl::StrCat(
        "ShmClient::ServerInfo: unexpected response type ", resp_type));
  }
  ServerInfoResponse resp;
  if (!resp.ParseFromString(resp_body)) {
    return absl::InternalError(
        "ShmClient::ServerInfo: malformed ServerInfoResponse");
  }
  table_info->clear();
  table_info->reserve(resp.table_info_size());
  for (const auto& info : resp.table_info()) {
    table_info->push_back(info);
  }
  return absl::OkStatus();
}

absl::Status ShmClient::MutatePriorities(
    const std::string& table, const std::vector<KeyWithPriority>& updates,
    const std::vector<uint64_t>& deletes) {
  // ticket ⑩: ride the INSERT flow. Hold conn_.insert_flow_mu across the whole
  // send→read-ACK sequence so this caller thread and RunShmWorker (the insert
  // worker background thread) are never both mid-flight on insert_c2s — that
  // would put two producers on one SPSC `head` and silently corrupt the ring.
  // The server-side dispatch is single-threaded and reads insert_c2s in order,
  // replying on insert_s2c in order, so whoever holds the mutex sends one
  // request and gets its matching ACK before releasing. Reuses the existing
  // reverb_service.proto MutatePrioritiesRequest (empty Response).
  MutatePrioritiesRequest req;
  req.set_table(table);
  for (const auto& u : updates) *req.add_updates() = u;
  for (uint64_t k : deletes) req.add_delete_keys(k);
  std::string body;
  req.SerializeToString(&body);

  absl::MutexLock lock(&conn_.insert_flow_mu);
  REVERB_RETURN_IF_ERROR(
      conn_.insert_c2s.Write(MUTATE_PRIORITIES, absl::MakeSpan(body)));

  MsgType resp_type;
  std::string resp_body;
  // ticket ⑩ 方向 C：控制面 ACK 等待用有限硬上限，避免 dispatch 被卡时 client
  // 无限忙等（mutate/reset 本身不含超时语义，靠此兜底）。
  REVERB_RETURN_IF_ERROR(
      ReadBlocking(&conn_.insert_s2c, &resp_type, &resp_body, conn_.control_fd,
                   kReadBlockingHardCap));

  if (resp_type == ERROR) {
    // mirror FetchOne's error mapping: NOT_FOUND -> NotFoundError (Python
    // FileNotFoundError), INVALID_ARGUMENT -> InvalidArgumentError, else
    // InternalError.
    ShmError err;
    if (!err.ParseFromString(resp_body)) {
      return absl::InternalError(
          "ShmClient::MutatePriorities: malformed ShmError");
    }
    if (err.code() == ShmError::NOT_FOUND) {
      return absl::NotFoundError(err.message());
    }
    if (err.code() == ShmError::INVALID_ARGUMENT) {
      return absl::InvalidArgumentError(err.message());
    }
    return absl::InternalError(
        absl::StrCat("ShmClient::MutatePriorities: server error: ", err.message()));
  }
  if (resp_type != MUTATE_ACK) {
    return absl::InternalError(absl::StrCat(
        "ShmClient::MutatePriorities: unexpected response type ", resp_type));
  }
  return absl::OkStatus();  // MUTATE_ACK is empty
}

absl::Status ShmClient::Reset(const std::string& table) {
  // ticket ⑩: same insert-flow round-trip as MutatePriorities (see above for
  // the mutex rationale). ResetRequest{table}; RESET_ACK is empty.
  ResetRequest req;
  req.set_table(table);
  std::string body;
  req.SerializeToString(&body);

  absl::MutexLock lock(&conn_.insert_flow_mu);
  REVERB_RETURN_IF_ERROR(
      conn_.insert_c2s.Write(RESET, absl::MakeSpan(body)));

  MsgType resp_type;
  std::string resp_body;
  // ticket ⑩ 方向 C：控制面 ACK 等待用有限硬上限（同 MutatePriorities）。
  REVERB_RETURN_IF_ERROR(
      ReadBlocking(&conn_.insert_s2c, &resp_type, &resp_body, conn_.control_fd,
                   kReadBlockingHardCap));

  if (resp_type == ERROR) {
    ShmError err;
    if (!err.ParseFromString(resp_body)) {
      return absl::InternalError("ShmClient::Reset: malformed ShmError");
    }
    if (err.code() == ShmError::NOT_FOUND) {
      return absl::NotFoundError(err.message());
    }
    if (err.code() == ShmError::INVALID_ARGUMENT) {
      return absl::InvalidArgumentError(err.message());
    }
    return absl::InternalError(
        absl::StrCat("ShmClient::Reset: server error: ", err.message()));
  }
  if (resp_type != RESET_ACK) {
    return absl::InternalError(absl::StrCat(
        "ShmClient::Reset: unexpected response type ", resp_type));
  }
  return absl::OkStatus();  // RESET_ACK is empty
}

absl::Status ShmClient::Checkpoint(std::string* path) {
  // ticket ⑪: same insert-flow round-trip as MutatePriorities/Reset (see
  // MutatePriorities for the insert_flow_mu rationale). CheckpointRequest is
  // empty; CheckpointResponse carries checkpoint_path. No table routing —
  // checkpoint is cross-table.
  CheckpointRequest req;
  std::string body;
  req.SerializeToString(&body);

  absl::MutexLock lock(&conn_.insert_flow_mu);
  REVERB_RETURN_IF_ERROR(
      conn_.insert_c2s.Write(CHECKPOINT, absl::MakeSpan(body)));

  MsgType resp_type;
  std::string resp_body;
  // ticket ⑪: same kReadBlockingHardCap cap as ⑩'s control-plane ACKs.
  REVERB_RETURN_IF_ERROR(
      ReadBlocking(&conn_.insert_s2c, &resp_type, &resp_body, conn_.control_fd,
                   kReadBlockingHardCap));

  if (resp_type == ERROR) {
    // Server maps no-checkpointer / Save failure to ShmError::INTERNAL.
    ShmError err;
    if (!err.ParseFromString(resp_body)) {
      return absl::InternalError(
          "ShmClient::Checkpoint: malformed ShmError");
    }
    return absl::InternalError(absl::StrCat(
        "ShmClient::Checkpoint: server error: ", err.message()));
  }
  if (resp_type != CHECKPOINT_RESP) {
    return absl::InternalError(absl::StrCat(
        "ShmClient::Checkpoint: unexpected response type ", resp_type));
  }
  CheckpointResponse resp;
  if (!resp.ParseFromString(resp_body)) {
    return absl::InternalError(
        "ShmClient::Checkpoint: malformed CheckpointResponse");
  }
  *path = resp.checkpoint_path();
  return absl::OkStatus();
}

absl::Status ShmClient::NewSampler(const std::string& table_name,
                                   const Sampler::Options& options,
                                   std::unique_ptr<ShmSampler>* sampler) {
  auto s = ShmSampler::Create(&conn_, table_name, options);
  REVERB_RETURN_IF_ERROR(s.status());
  *sampler = std::move(*s);
  return absl::OkStatus();
}

absl::Status ShmClient::NewTrajectoryWriter(
    const TrajectoryWriter::Options& options,
    std::unique_ptr<TrajectoryWriter>* writer) {
  REVERB_RETURN_IF_ERROR(options.Validate());
  // SHM mode: the writer's RunShmWorker sends inserts over conn_. No local
  // tables — the server owns the Table. ticket ⑧ step 2: populate
  // flat_signature_map from a live SERVER_INFO round-trip so that
  // CreateItem's ItemAndRefs::Validate runs the same signature check as
  // gRPC/LocalClient. Each table occupies one entry; a table with no
  // signature gets nullopt (Validate skips it). A table absent from the
  // response is treated as "unknown table" by Validate.
  TrajectoryWriter::Options effective_options = options;
  std::vector<TableInfo> server_info;
  REVERB_RETURN_IF_ERROR(ServerInfo(&server_info));
  internal::FlatSignatureMap signatures;
  for (const auto& info : server_info) {
    internal::DtypesAndShapes& entry = signatures[info.name()];
    REVERB_RETURN_IF_ERROR(
        internal::FlatSignatureFromTableInfo(info, &entry));
  }
  effective_options.flat_signature_map = std::move(signatures);
  *writer = std::make_unique<TrajectoryWriter>(&conn_, effective_options);
  return absl::OkStatus();
}

absl::Status ShmClient::NewStructuredWriter(
    std::vector<StructuredWriterConfig> configs,
    std::unique_ptr<StructuredWriter>* writer) {
  // ponytail: 主体收敛到 MakeStructuredWriter(与 Client/InProcessClient 共用),
  // 仅 NewTrajectoryWriter 钩子为本客户端专有(走 SHM live SERVER_INFO 路径)。
  return MakeStructuredWriter(
      std::move(configs), writer,
      [this](const TrajectoryWriter::Options& options,
             std::unique_ptr<TrajectoryWriter>* trajectory_writer) {
        return NewTrajectoryWriter(options, trajectory_writer);
      });
}

absl::Status ShmClient::NewWriter(int /*chunk_length*/,
                                  int /*max_timesteps*/,
                                  bool /*delta_encoded*/,
                                  int /*max_in_flight_items*/,
                                  std::unique_ptr<Writer>* /*writer*/) {
  // ponytail: won't fix — the plain Writer (writer.h) has no SHM transport
  // seam. Its local ctor takes a tables map the SHM client doesn't hold.
  // Adding SHM to Writer would duplicate RunShmWorker for a legacy API.
  // The Python ShmClient overrides `writer`/`insert` to raise
  // NotImplementedError before reaching here. Use NewTrajectoryWriter /
  // NewStructuredWriter for SHM inserts.
  return absl::UnimplementedError(
      "ShmClient::NewWriter is not implemented for SHM; use "
      "NewTrajectoryWriter or NewStructuredWriter.");
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
