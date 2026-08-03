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

#include "reverb/cc/shm/shm_connection.h"

#include <poll.h>
#include <sched.h>
#include <sys/socket.h>  // recv, MSG_PEEK
#include <unistd.h>

namespace deepmind {
namespace reverb {
namespace shm {

absl::Status WriteBlocking(Ring* ring, MsgType msg_type,
                           absl::Span<const char> payload, int control_fd,
                           absl::Duration timeout) {
  absl::Time deadline = absl::Now() + timeout;
  while (true) {
    absl::Status s = ring->TryWrite(msg_type, payload);
    if (s.ok()) {
      // ticket 03: wake the server if it flagged itself asleep on THIS ring.
      // seq_cst load pairs with the server's seq_cst store before its
      // re-check (see RingHeader::server_asleep): reading 0 guarantees the
      // server sees this write in its re-check and never blocks on it.
      // Reading 1 costs one DONTWAIT send; errors are ignored — a lost byte
      // only falls back to the server's 50ms poll timeout, and a dead peer
      // is reported by the liveness probe on the next failed pass. Hot path
      // (flag 0, server awake) adds ZERO syscalls.
      if (control_fd >= 0 &&
          ring->server_asleep()->load(std::memory_order_seq_cst) != 0) {
        char b = 0;
        (void)send(control_fd, &b, 1, MSG_DONTWAIT | MSG_NOSIGNAL);
      }
      return absl::OkStatus();
    }
    if (!absl::IsResourceExhausted(s)) return s;  // permanent error, no retry
    if (control_fd >= 0 && IsPeerClosed(control_fd)) {
      return absl::UnavailableError("SHM peer closed connection");
    }
    if (absl::Now() >= deadline) {
      return absl::DeadlineExceededError("SHM ring write timed out");
    }
    sched_yield();
  }
}

bool IsPeerClosed(int fd) {
  if (fd < 0) return false;
  // poll with zero timeout: POLLHUP/POLLERR => peer closed. POLLIN then needs
  // disambiguation (data vs. a clean 0-byte EOF). A healthy idle peer has the
  // fd open and nothing to send => poll returns 0 (alive).
  struct pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLIN;
  pfd.revents = 0;
  int n = poll(&pfd, 1, 0);
  if (n <= 0) return false;  // no event (or EINTR) => alive
  if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) return true;
  if (pfd.revents & POLLIN) {
    // Data or EOF; peek 1 byte to disambiguate. recv()==0 is EOF (clean
    // close). r>0 means stray bytes are readable (MSG_PEEK keeps them; they
    // may re-trigger POLLIN but that is harmless — the peer stays alive).
    char buf;
    ssize_t r = recv(fd, &buf, 1, MSG_PEEK);
    if (r == 0) return true;  // peer closed
  }
  return false;
}

void ShmConnection::Close() {
  closed.store(true, std::memory_order_release);
  // ticket ⑥: closing the client's liveness fd is the crash/close signal the
  // server poll()s (spec §8.8). -1 means the server side (which stores its fd
  // in ClientState.fd) or a moved-from object.
  if (control_fd >= 0) close(control_fd);
  control_fd = -1;
}

ShmConnection::~ShmConnection() { Close(); }

ShmConnection::ShmConnection(ShmConnection&& other) noexcept
    : insert_c2s(std::move(other.insert_c2s)),
      insert_s2c(std::move(other.insert_s2c)),
      sample_c2s(std::move(other.sample_c2s)),
      sample_s2c(std::move(other.sample_s2c)),
      pool(std::move(other.pool)),
      pool_shm_name(std::move(other.pool_shm_name)),
      control_fd(other.control_fd),
      closed(other.closed.load(std::memory_order_acquire)) {
  // insert_flow_mu is non-movable (absl::Mutex); leave this instance's mutex
  // default-constructed (unlocked). A ShmConnection is moved exactly once
  // before any worker thread starts, so the destination's mutex is the one
  // RunShmWorker/MutatePriorities/Reset all see. See shm_connection.h.
  other.control_fd = -1;  // stolen, so ~other does not double-close
}

ShmConnection& ShmConnection::operator=(ShmConnection&& other) noexcept {
  if (this != &other) {
    insert_c2s = std::move(other.insert_c2s);
    insert_s2c = std::move(other.insert_s2c);
    sample_c2s = std::move(other.sample_c2s);
    sample_s2c = std::move(other.sample_s2c);
    pool = std::move(other.pool);
    pool_shm_name = std::move(other.pool_shm_name);
    if (control_fd >= 0) close(control_fd);
    control_fd = other.control_fd;
    other.control_fd = -1;
    closed.store(other.closed.load(std::memory_order_acquire),
                 std::memory_order_release);
    // insert_flow_mu intentionally NOT moved (non-movable; see move ctor).
  }
  return *this;
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
