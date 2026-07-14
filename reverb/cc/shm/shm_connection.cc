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
#include <sys/socket.h>  // recv, MSG_PEEK
#include <unistd.h>

namespace deepmind {
namespace reverb {
namespace shm {

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

ShmConnection::~ShmConnection() {
  // ticket ⑥: closing the client's liveness fd is the crash/close signal the
  // server poll()s (spec §8.8). -1 means the server side (which stores its fd
  // in ClientState.fd) or a moved-from object.
  if (control_fd >= 0) close(control_fd);
}

ShmConnection::ShmConnection(ShmConnection&& other) noexcept
    : insert_c2s(std::move(other.insert_c2s)),
      insert_s2c(std::move(other.insert_s2c)),
      sample_c2s(std::move(other.sample_c2s)),
      sample_s2c(std::move(other.sample_s2c)),
      pool(std::move(other.pool)),
      pool_shm_name(std::move(other.pool_shm_name)),
      control_fd(other.control_fd) {
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
  }
  return *this;
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
