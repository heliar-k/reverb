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

#include <unistd.h>

namespace deepmind {
namespace reverb {
namespace shm {

ShmConnection::~ShmConnection() {
  // ticket ⑥: closing the client's liveness fd is the crash/close signal the
  // server poll()s (spec §8.8). -1 means the server side (which stores its fd
  // in ClientState.fd) or a moved-from object.
  if (control_fd >= 0) close(control_fd);
}

ShmConnection::ShmConnection(ShmConnection&& other) noexcept
    : c2s(std::move(other.c2s)),
      s2c(std::move(other.s2c)),
      pool(std::move(other.pool)),
      pool_shm_name(std::move(other.pool_shm_name)),
      control_fd(other.control_fd) {
  other.control_fd = -1;  // stolen, so ~other does not double-close
}

ShmConnection& ShmConnection::operator=(ShmConnection&& other) noexcept {
  if (this != &other) {
    c2s = std::move(other.c2s);
    s2c = std::move(other.s2c);
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
