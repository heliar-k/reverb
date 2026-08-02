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

#include <atomic>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
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

// review #2 (ring-write-liveness): blocking-write helper mirroring the read
// side's ReadBlocking. Polls TryWrite with sched_yield until space is free;
// each pass probes `control_fd` for peer death (UnavailableError) and gives
// up at `timeout` (DeadlineExceededError). Without this, Ring::Write's bare
// sched_yield loop spins FOREVER at 100% CPU when the server's dispatch
// thread is wedged (e.g. a slow checkpoint, review #3) or dead — hanging
// TrajectoryWriter::Close()/GC with no error surfaced.
//
// `timeout` defaults to 60s, mirroring the read side's kReadBlockingHardCap:
// a robustness ceiling, not a semantic deadline — callers treat it as a
// transport error. control_fd < 0 disables the liveness probe (deadline
// still applies).
constexpr absl::Duration kWriteBlockingHardCap = absl::Seconds(60);
absl::Status WriteBlocking(Ring* ring, MsgType msg_type,
                           absl::Span<const char> payload, int control_fd,
                           absl::Duration timeout = kWriteBlockingHardCap);

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

  // ticket「shm-close-while-in-flight」: set by Close()/destructor BEFORE
  // control_fd is closed. In-flight read loops (ReadBlocking in shm_client.cc,
  // read_blocking in trajectory_writer.cc) poll this flag so a connection
  // closed underneath them errors out (UnavailableError) instead of spinning
  // forever on rings nobody serves. The fd probe alone is insufficient:
  // after close the fd is -1 (or recycled), and the probe is skipped for
  // fd < 0 — that hole hung DisconnectWithInFlightCallbacksDoesNotUaf.
  std::atomic<bool> closed{false};

  // Idempotent. Sets `closed` first (release), then closes control_fd.
  // In-flight workers observe `closed` and fail fast; the server observes
  // the fd EOF and reclaims the client (spec §8.8).
  void Close();

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
  //
  // ticket ⑩ 死锁复盘（2026-07-17）：此 mutex 曾被静态分析怀疑为偶发死锁根因
  // （锁范围覆盖 read ACK），但 gdb 抓栈证明无辜——真根因是服务端单线程
  // dispatch 在 HandleSample 的 rate-limiter 无限阻塞（队头阻塞），已由方向 A
  // （HandleSample 异步化）修复，见 shm_server.cc / tickets.md ⑩。此 mutex 保持
  // 原样：序列化整个 round-trip 反而防止两个生产者同时写 insert_c2s，是正确的。
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
