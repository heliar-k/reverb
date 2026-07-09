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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
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
// of `outstanding_offsets_` and `outbox` (single-threaded dispatch, decision
// A1/R11), so they need no lock in practice; `outbox_mu` is retained to match
// the spec §3.4 declaration and to future-proof a per-client dispatch split.
//
// ponytail: v1 dispatch is single-threaded; the mutex is uncontended. Upgrade
// path: per-client dispatch threads (spec §8.7 note) would contend here.
struct ClientState {
  int fd = -1;                       // udsocket fd (closed on disconnect)
  int client_pid = 0;
  ShmConnection conn;                // c2s, s2c rings; pool unused server-side
  absl::flat_hash_set<uint64_t> outstanding_offsets_;  // C3: crash recovery (⑥)

  // Stashed S→C messages that did not fit (ring full). Flushed each dispatch
  // pass via a non-blocking Write. ponytail: vector, upgrade to ring-buffer.
  absl::Mutex outbox_mu;
  std::vector<std::pair<uint16_t, std::string>> outbox
      ABSL_GUARDED_BY(outbox_mu);
};

// ShmServer owns ONE real Table (ponytail: multi-table later), a ShmBytePool
// (sole allocator, C4), and a bootstrap udsocket. A single dispatch thread
// polls the listen socket for new clients, then non-blocking-reads each
// client's C→S ring: SAMPLE → Table::Sample → UnpackChunkColumnAndSlice ON THE
// DISPATCH THREAD (A1) → memcpy result bytes into the pool (refcount=1, C3) →
// write SAMPLE_RESP to S→C (non-blocking, stashed in outbox if full, §8.7);
// RELEASE → Unref each offset, →0 deallocates. Insert is NOT wired (ticket ④).
class ShmServer {
 public:
  // Create the pool + bootstrap server. `socket_path` is the udsocket path.
  // ponytail: ONE table per server for v1; multi-table later.
  static absl::StatusOr<std::unique_ptr<ShmServer>> Create(
      std::shared_ptr<Table> table, const std::string& socket_path);

  ~ShmServer();

  ShmServer(const ShmServer&) = delete;
  ShmServer& operator=(const ShmServer&) = delete;

  // Launch the dispatch thread.
  absl::Status Start();

  // Stop the dispatch thread, clean up clients, unlink SHM segments.
  void Stop();

  const std::string& socket_path() const { return socket_path_; }

 private:
  ShmServer(std::shared_ptr<Table> table, std::string socket_path,
            ShmBytePool pool, ShmBootstrapServer bootstrap);

  // dispatch thread main loop
  void DispatchLoop();

  // Accept a waiting client (non-blocking via poll on listen fd). Returns true
  // if a client was accepted.
  bool TryAccept();

  // Drain one client's C→S ring (non-blocking Read), dispatching each request.
  void HandleClientRequests(size_t client_id);

  // Flush the client's outbox with non-blocking S→C writes (§8.7).
  void FlushOutbox(ClientState& state);

  // Sample path: Table::Sample → unpack (A1) → pool memcpy (C3) → SAMPLE_RESP.
  absl::Status HandleSample(ClientState& state, const ShmSampleRequest& req);

  // Release path: Unref each offset, →0 deallocates (C3).
  absl::Status HandleRelease(ClientState& state,
                             const ShmReleaseRequest& req);

  // Enqueue a S→C message: try a non-blocking write, stash in outbox if full.
  absl::Status EnqueueS2C(ClientState& state, MsgType type,
                          absl::string_view body);

  std::shared_ptr<Table> table_;
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
