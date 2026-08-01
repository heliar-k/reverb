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

#ifndef REVERB_CC_SHM_SHM_SERVER_H_
#define REVERB_CC_SHM_SHM_SERVER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/platform/default/hash_map.h"
#include "reverb/cc/platform/default/hash_set.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/shm/shm_protocol.pb.h"
#include "reverb/cc/support/task_executor.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {
namespace shm {

// Per-client state held by ShmServer. The dispatch thread is the sole mutator
// of `outstanding_offsets_` and the outboxes (single-threaded dispatch,
// decision A1/R11), so they need no lock in practice; the mutexes are retained
// to match the spec §3.4 declaration and to future-proof a per-client dispatch
// split. Decision D: outboxes are per-flow (insert/sample) so each s2c ring's
// retry queue stays homogeneous.
//
// ponytail: v1 dispatch is single-threaded; the mutexes are uncontended.
// Upgrade path: per-client dispatch threads (spec §8.7 note) would contend.
struct ClientState {
  int fd = -1;                       // udsocket fd (closed on disconnect)
  int client_pid = 0;
  ShmConnection conn;  // 4 rings (insert+sample c2s/s2c); pool unused server-side
  internal::flat_hash_set<uint64_t> outstanding_offsets_;  // C3: crash recovery (⑥)

  // Stashed S→C messages that did not fit (ring full). Flushed each dispatch
  // pass via a non-blocking Write. Decision D: separate per-flow outboxes so
  // a full insert s2c ring does not block sample responses (and vice versa).
  // ponytail: vector, upgrade to ring-buffer.
  absl::Mutex insert_outbox_mu;
  std::vector<std::pair<uint16_t, std::string>> insert_outbox
      ABSL_GUARDED_BY(insert_outbox_mu);
  absl::Mutex sample_outbox_mu;
  std::vector<std::pair<uint16_t, std::string>> sample_outbox
      ABSL_GUARDED_BY(sample_outbox_mu);

  // Insert-callback keepalive (ticket ④): InsertOrAssignAsync stores a
  // weak_ptr to the callback; the table worker fires it asynchronously, AFTER
  // HandleInsert returns. The shared_ptr must therefore outlive HandleInsert.
  // We stash them here and clear them once the aggregate InsertAck is enqueued
  // (i.e. the last item's callback has fired). Guarded by insert_outbox_mu
  // (the callback fires on the table callback-executor thread, not the
  // dispatch thread). Mirrors Writer::WritePendingDataLocal's
  // local_pending_callbacks_.
  std::vector<std::shared_ptr<Table::InsertCallback>> pending_insert_callbacks
      ABSL_GUARDED_BY(insert_outbox_mu);

  // ticket ⑩ 死锁修复（方向 A）：异步 sample 的完成回调在 table worker 线程
  // 触发，不能直接碰 pool_/outstanding_offsets_（单线程 dispatch 不变式）。
  // 回调把 SampledItem + 路由元数据攒进这个 mutex 保护的队列，dispatch 线程
  // 每轮 DrainPendingSamples 取出做 unpack+pool+写 SAMPLE_RESP。镜像
  // pending_insert_callbacks 的 callback→dispatch-drain 模式。
  struct PendingSample {
    ShmSampleRequest req;          // 原始请求（table 名 + 超时映射在 status 里）
    absl::Status status;           // table worker 的结果（含 DeadlineExceeded）
    Table::SampledItem item;       // 成功时的采样项
  };
  absl::Mutex pending_samples_mu;
  std::vector<PendingSample> pending_samples ABSL_GUARDED_BY(pending_samples_mu);
  // EnqueSampleRequest 存 weak_ptr<SamplingCallback>，table worker 在
  // HandleSample 返回后触发回调，shared_ptr 必须存活到回调触发。存这里；回调
  // 触发时按裸指针 key 自清 erase（裸指针存在堆上 shared_ptr 控制块里按值捕获，
  // 避免局部变量悬空）。ponytail: O(n) scan erase，n=in-flight sample 数（<=8）。
  // Ceil: 高并发可改 hash_set 按指针查。Upgrade: 同。
  std::vector<std::shared_ptr<Table::SamplingCallback>> pending_sample_callbacks
      ABSL_GUARDED_BY(pending_samples_mu);

  // ticket ⑥: set when the client sends an explicit CLOSE. The dispatch loop's
  // IsClientDead check then routes it through HandleDisconnect next pass.
  bool close_requested = false;
};

// ShmServer owns ALL tables (ticket ⑨: routed by table name) keyed by name,
// a ShmBytePool (sole allocator, C4), and a bootstrap udsocket. A single
// dispatch thread polls the listen socket for new clients, then non-blocking-
// reads each client's C→S ring: SAMPLE → FindTable(req.table) →
// Table::EnqueSampleRequest (ASYNC, ticket ⑩ 死锁修复：dispatch 不阻塞于 rate
// limiter) → 完成回调攒进 pending_samples → DrainPendingSamples 在 dispatch
// 线程做 UnpackChunkColumnAndSlice → memcpy result bytes into the pool
// (refcount=1, C3) → write SAMPLE_RESP to S→C (non-blocking, stashed in
// outbox if full, §8.7); RELEASE → Unref each offset, →0 deallocates. Insert
// routes each PrioritizedItem to its named table via FindTable (ticket ④+⑨).
class ShmServer {
 public:
  // Create the pool + bootstrap server. `socket_path` is the udsocket path.
  // `tables` must be non-empty with unique names (validated here; the Python
  // `Server` also checks, but C++ defends itself). `checkpointer` is optional
  // (ticket ⑪): when provided, HandleCheckpoint saves all tables and returns
  // the path; when null, HandleCheckpoint returns FailedPreconditionError
  // (mirrors InProcessClient::Checkpoint).
  static absl::StatusOr<std::unique_ptr<ShmServer>> Create(
      std::vector<std::shared_ptr<Table>> tables,
      const std::string& socket_path,
      std::shared_ptr<Checkpointer> checkpointer = nullptr);

  ~ShmServer();

  ShmServer(const ShmServer&) = delete;
  ShmServer& operator=(const ShmServer&) = delete;

  // Launch the dispatch thread.
  absl::Status Start();

  // Stop the dispatch thread, stop all tables (Table::Stop: Close + join
  // worker + drain callback executor — guarantees no table callback can fire
  // afterwards, review #1), then clean up clients and unlink SHM segments.
  void Stop();

  const std::string& socket_path() const { return socket_path_; }

  // ticket ⑥ test-only: close the accepted udsocket fd of client 0 WITHOUT
  // running HandleDisconnect on the dispatch thread. This simulates the server
  // side of the connection dropping (server crash / fd close) so the CLIENT's
  // liveness control_fd sees EOF — the path ReadBlocking must detect to fail
  // fast. Stop() can't be used for this because it joins the dispatch thread,
  // which may be blocked in Table::Sample; closing the fd here lets the test
  // observe the client's EOF reaction deterministically. No-op if no client.
  void CloseClientFdForTest();

 private:
  ShmServer(std::vector<std::shared_ptr<Table>> tables, std::string socket_path,
            ShmBytePool pool, ShmBootstrapServer bootstrap,
            std::shared_ptr<Checkpointer> checkpointer,
            std::string name_token);

  // dispatch thread main loop
  void DispatchLoop();

  // Accept a waiting client (non-blocking via poll on listen fd). Returns true
  // if a client was accepted.
  bool TryAccept();

  // Drain one client's insert C→S ring (non-blocking Read), dispatching each
  // request (ALLOCATE/INSERT/RELEASE).
  void HandleInsertRequests(size_t client_id);

  // Drain one client's sample C→S ring (non-blocking Read), dispatching each
  // request (SAMPLE/RELEASE). Decision D: insert and sample flows have
  // separate c2s rings, drained independently so neither blocks the other.
  void HandleSampleRequests(size_t client_id);

  // Flush both per-flow outboxes with non-blocking S→C writes (§8.7).
  void FlushOutbox(ClientState& state);

  // Sample path (ticket ⑩ 死锁修复，方向 A): 异步入队 Table::EnqueSampleRequest，
  // dispatch 不阻塞于 rate limiter。完成回调在 table worker 线程把 SampledItem
  // 攒进 ClientState::pending_samples；DrainPendingSamples 在 dispatch 线程做
  // unpack+pool+SAMPLE_RESP（保持 pool_/outstanding_offsets_ 单线程不变式）。
  // 未知表仍同步返 ERROR（不入队）。
  // review #1: takes shared_ptr so the completion callback can capture it —
  // the ClientState then outlives clients_.erase()/clear() until the
  // callback returns (HandleDisconnect path, where the table can't be
  // stopped per-client).
  absl::Status HandleSample(std::shared_ptr<ClientState> state,
                            const ShmSampleRequest& req);

  // 取出异步完成的 sample，在 dispatch 线程做 unpack+pool memcpy+写 SAMPLE_RESP
  // （或写 ERROR on 失败/超时）。镜像 HandleInsert 的 callback→outbox 模式。
  void DrainPendingSamples(ClientState& state);

  // Release path: Unref each offset, →0 deallocates (C3).
  absl::Status HandleRelease(ClientState& state,
                             const ShmReleaseRequest& req);

  // Insert path (ticket ④): deserialize each ShmChunkRef's ChunkData from the
  // pool, build a TableItem per PrioritizedItem, InsertOrAssignAsync. The
  // table-worker completion callback stashes InsertAck{keys,
  // offsets_to_release} into the client's outbox (C2); the dispatch thread —
  // the sole S→C producer — drains it via FlushOutbox.
  // review #1: shared_ptr param, see HandleSample.
  absl::Status HandleInsert(std::shared_ptr<ClientState> state,
                            const ShmInsertRequest& req);

  // C4 allocate path: client requests a pool offset; server (sole allocator)
  // grants it. Mirrors byte_pool_echo_test's inline handler.
  absl::Status HandleAllocate(ClientState& state,
                              const ShmAllocateRequest& req);

  // ticket ⑩: control-plane handlers. These ride the INSERT flow (insert_c2s
  // → insert_s2c) so they reuse EnqueueInsertS2C. The client serializes them
  // against RunShmWorker via ShmConnection::insert_flow_mu (see shm_client.cc)
  // — the server side is single-threaded dispatch, so no extra server lock.
  // MutatePriorities reuses the existing reverb_service.proto request type;
  // maps FindTable miss → ShmError::NOT_FOUND (client raises FileNotFoundError).
  absl::Status HandleMutatePriorities(ClientState& state,
                                     const MutatePrioritiesRequest& req);
  absl::Status HandleReset(ClientState& state, const ResetRequest& req);

  // ticket ⑪: checkpoint all tables via the injected checkpointer. Mirrors
  // InProcessClient::Checkpoint / ReverbServiceImpl::Checkpoint — gathers all
  // tables_, calls checkpointer_->Save(tables, keep_latest=1, path), returns
  // the path in CheckpointResponse (CHECKPOINT_RESP on the insert s2c flow).
  // No checkpointer -> FailedPreconditionError as ShmError::INTERNAL (mirrors
  // InProcessClient). Checkpoint is cross-table, so no FindTable routing.
  // review #3 (dispatch-thread-hol): Save is UNBOUNDED disk I/O and must not
  // run on the single dispatch thread — it is scheduled on
  // checkpoint_executor_ and the response rides back via the client's
  // insert_outbox (the callback→dispatch-drain pattern, same as async
  // inserts). shared_ptr param keeps the ClientState alive through the
  // out-of-band response (review #1 pattern).
  absl::Status HandleCheckpoint(std::shared_ptr<ClientState> state);

  // ticket ⑧ step 2: on-demand server_info round-trip. Gathers each table's
  // live TableInfo (current_size, signature, etc.) into a ServerInfoResponse
  // and returns it as SERVER_INFO_RESP on the insert s2c flow. Replaces the
  // step-1 bootstrap snapshot so server_info() reflects mid-session state.
  // Cross-table, always succeeds — no FindTable routing, no error path.
  absl::Status HandleServerInfo(ClientState& state);

  // Enqueue a S→C message on the INSERT flow's s2c ring: try a non-blocking
  // write, stash in insert_outbox if full.
  absl::Status EnqueueInsertS2C(ClientState& state, MsgType type,
                                absl::string_view body);
  // Enqueue a S→C message on the SAMPLE flow's s2c ring.
  absl::Status EnqueueSampleS2C(ClientState& state, MsgType type,
                                absl::string_view body);

  // ticket ⑥: detect a dead client (crash or graceful close) by probing its
  // udsocket fd for POLLHUP/POLLERR/EOF without blocking. Returns true if the
  // client's fd is closed. The fd is the liveness signal kept open for the
  // connection lifetime (ShmConnection::control_fd on the client side).
  bool IsClientDead(const ClientState& state);

  // ticket ⑥: reclaim a crashed/closed client's resources — centralized offset
  // release (C3), shm_unlink its two rings, close fd, erase from clients_.
  // Called from the dispatch loop on EOF/HUP and on an explicit CLOSE.
  void HandleDisconnect(size_t client_id);

  // Per-client cleanup (shared by HandleDisconnect and Stop): ReleaseAll the
  // outstanding offsets, shm_unlink the two rings, close the fd. Does NOT
  // erase from clients_ (the caller does). `unlink_rings` is false on Stop so
  // Stop is idempotent with the Ring destructor's own owner-unlink.
  void CleanupClient(ClientState& state, bool unlink_rings);

  // ticket ⑨: shared table-name routing. Returns the named table or
  // NotFoundError. ⑩ (mutate_priorities/reset) reuses this seam.
  absl::StatusOr<std::shared_ptr<Table>> FindTable(
      const std::string& name) const;

  // ponytail: map-only, no parallel ordered list. server_info fills
  // table_info by iterating `tables_`; hash_map order is unspecified but the
  // client consumes server_info into a name→TableInfo dict (set-equality in
  // parity tests), so ordering is irrelevant. Collapse to nothing smaller;
  // add an ordered vector only if a test starts asserting table_info order.
  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables_;
  std::string socket_path_;
  // ticket #7: socket path + per-server epoch, used as the token for all SHM
  // segment names (pool + per-client rings). A crash-restarted server on the
  // same socket path gets a fresh epoch, so it never collides with — and
  // unlink-and-retries out from under — a live client's segments.
  std::string name_token_;
  ShmBytePool pool_;
  ShmBootstrapServer bootstrap_;
  // ticket ⑪: optional, injected at Create. nullptr when the Python Server
  // constructs ShmServer without one (shouldn't happen in practice — Server
  // always builds a default checkpointer — but C++ defends itself).
  std::shared_ptr<Checkpointer> checkpointer_;
  // review #1 (shm-clientstate-lifetime): shared_ptr so insert/sample
  // completion callbacks (fired on table callback-executor threads, capturing
  // the ClientState) keep the state alive until they return — erase()/clear()
  // only drop the server's reference. The Stop()-path window is additionally
  // closed deterministically by stopping tables before clearing clients.
  std::vector<std::shared_ptr<ClientState>> clients_;

  std::thread dispatch_thread_;
  std::atomic<bool> running_{false};

  // review #3: single-thread executor for checkpoint Saves — keeps unbounded
  // disk I/O off the dispatch thread. Declared after tables_/clients_ so it
  // is destroyed BEFORE them (its queued tasks reference both). Closed
  // (drained + joined) in Stop() before tables stop; ~TaskExecutor re-enters
  // Close() safely.
  TaskExecutor checkpoint_executor_{1, "ShmCheckpointExecutor"};
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_SERVER_H_
