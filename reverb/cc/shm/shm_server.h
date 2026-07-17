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
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/shm/shm_protocol.pb.h"
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

  // ticket ⑥: set when the client sends an explicit CLOSE. The dispatch loop's
  // IsClientDead check then routes it through HandleDisconnect next pass.
  bool close_requested = false;
};

// ShmServer owns ALL tables (ticket ⑨: routed by table name) keyed by name,
// a ShmBytePool (sole allocator, C4), and a bootstrap udsocket. A single
// dispatch thread polls the listen socket for new clients, then non-blocking-
// reads each client's C→S ring: SAMPLE → FindTable(req.table) →
// Table::Sample → UnpackChunkColumnAndSlice ON THE DISPATCH THREAD (A1) →
// memcpy result bytes into the pool (refcount=1, C3) → write SAMPLE_RESP to
// S→C (non-blocking, stashed in outbox if full, §8.7); RELEASE → Unref each
// offset, →0 deallocates. Insert routes each PrioritizedItem to its named
// table via FindTable (ticket ④ + ⑨).
class ShmServer {
 public:
  // Create the pool + bootstrap server. `socket_path` is the udsocket path.
  // `tables` must be non-empty with unique names (validated here; the Python
  // `Server` also checks, but C++ defends itself).
  static absl::StatusOr<std::unique_ptr<ShmServer>> Create(
      std::vector<std::shared_ptr<Table>> tables,
      const std::string& socket_path);

  ~ShmServer();

  ShmServer(const ShmServer&) = delete;
  ShmServer& operator=(const ShmServer&) = delete;

  // Launch the dispatch thread.
  absl::Status Start();

  // Stop the dispatch thread, clean up clients, unlink SHM segments.
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
            ShmBytePool pool, ShmBootstrapServer bootstrap);

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

  // Sample path: Table::Sample → unpack (A1) → pool memcpy (C3) → SAMPLE_RESP.
  absl::Status HandleSample(ClientState& state, const ShmSampleRequest& req);

  // Release path: Unref each offset, →0 deallocates (C3).
  absl::Status HandleRelease(ClientState& state,
                             const ShmReleaseRequest& req);

  // Insert path (ticket ④): deserialize each ShmChunkRef's ChunkData from the
  // pool, build a TableItem per PrioritizedItem, InsertOrAssignAsync. The
  // table-worker completion callback stashes InsertAck{keys,
  // offsets_to_release} into the client's outbox (C2); the dispatch thread —
  // the sole S→C producer — drains it via FlushOutbox.
  absl::Status HandleInsert(ClientState& state, const ShmInsertRequest& req);

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
  ShmBytePool pool_;
  ShmBootstrapServer bootstrap_;
  std::vector<std::unique_ptr<ClientState>> clients_;

  std::thread dispatch_thread_;
  std::atomic<bool> running_{false};
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_SERVER_H_
