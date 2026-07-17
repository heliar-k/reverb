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

#ifndef REVERB_CC_SHM_SHM_CONNECTION_H_
#define REVERB_CC_SHM_SHM_CONNECTION_H_

#include <string>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"

namespace deepmind {
namespace reverb {
namespace shm {

// ticket ⑥ (spec §8.8): returns true if the peer on the OTHER end of `fd`
// has closed the connection (EOF / POLLHUP / POLLERR). Used on BOTH sides:
//   - server: probe ClientState.fd to detect a crashed/gone client.
//   - client: probe ShmConnection::control_fd to detect a gone server so a
//     ReadBlocking poll waiting for an S→C response fails fast instead of
//     spinning forever.
// fd < 0 (no fd, e.g. moved-from / server side) => false (nothing to probe).
bool IsPeerClosed(int fd);

// The SPSC rings wired between a server and one client, plus the shared
// byte pool. Decision D (per-flow rings): there are TWO ring PAIRS, one for
// the insert flow (TrajectoryWriter's RunShmWorker) and one for the sample
// flow (ShmSampler's worker). Each pair is strict SPSC: the flow's single
// client worker thread is the sole producer on its c2s ring and sole consumer
// on its s2c ring. Splitting the pairs lets the two worker threads run
// concurrently WITHOUT a mutex — the single-pair design violated the SPSC
// invariant once both writers started background threads (two producers on one
// c2s `head`, no CAS => data corruption).
//
// The `pool` handle is only meaningful on the client side (where it is
// `ShmBytePool::Open`'d read/write per decision C4); the server keeps its own
// `ShmBytePool` (the owner/allocator) inside `ShmServer` and does not share it
// through this struct. Both sides read sample bytes via `pool.At(offset)`.
//
// `control_fd` (ticket ⑥): the open udsocket fd kept for the connection
// lifetime as a liveness signal. The SERVER stores its accepted fd in
// `ClientState.fd` (not here) and leaves this -1; the CLIENT stores its
// bootstrap fd here and ~ShmConnection closes it. When the client process
// crashes or ~ShmClient runs, the fd closes -> the server's poll() sees
// POLLHUP/EOF -> HandleDisconnect (spec §8.8).
//
// ponytail: a plain struct with a destructor closing control_fd, no factory.
// Move-only (Ring/ShmBytePool are move-only). The move-ctor must steal
// control_fd and null the source so ~ShmConnection does not double-close.
struct ShmConnection {
  Ring insert_c2s;  // client insert worker -> server (ALLOCATE/INSERT/RELEASE)
  Ring insert_s2c;  // server -> client insert worker (ALLOCATE_RESP/INSERT_ACK)
  Ring sample_c2s;  // client sample worker -> server (SAMPLE/RELEASE)
  Ring sample_s2c;  // server -> client sample worker (SAMPLE_RESP)
  ShmBytePool pool;  // client-side RW mapping (C4); server keeps its own
  std::string pool_shm_name;
  int control_fd = -1;  // client liveness fd (ticket ⑥); -1 = none

  ShmConnection() = default;
  ~ShmConnection();
  ShmConnection(ShmConnection&& other) noexcept;
  ShmConnection& operator=(ShmConnection&& other) noexcept;
  ShmConnection(const ShmConnection&) = delete;
  ShmConnection& operator=(const ShmConnection&) = delete;

  // ticket ⑩: serializes the send-request → read-ACK round-trip on the
  // INSERT flow so two producers never touch insert_c2s's single `head` at
  // once. Both RunShmWorker (the insert worker background thread: its
  // ALLOCATE→ALLOCATE_RESP and INSERT→INSERT_ACK round-trips) AND the
  // caller-thread MutatePriorities/Reset round-trips acquire this mutex for
  // the whole send→read sequence. The sample flow
  // (sample_c2s/sample_s2c, ShmSampler's worker) is untouched and stays
  // lock-free — the mutex only contends when a control-plane call overlaps
  // an in-flight insert, which is rare.
  // ponytail: one mutex on the existing insert flow instead of a third
  // dedicated control ring pair. Ceiling: control-plane ops (mutate/reset)
  // serialize against in-flight inserts on this client; the hot sample path
  // is unaffected. Upgrade path: a third SPSC ring pair dedicated to
  // control-plane traffic would let mutate/reset run fully concurrent with
  // inserts if profiling shows this mutex contending.
  // Note: absl::Mutex is non-movable, so the move ctor/assignment below leave
  // the destination's mutex default-constructed (unlocked) and do not steal
  // the source's. A connection is moved exactly once (ShmClient::Connect into
  // the ShmClient member, or TryAccept into ClientState) before any worker
  // thread starts, so the post-move mutex is the one all threads see.
  mutable absl::Mutex insert_flow_mu;
};

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_SHM_CONNECTION_H_
