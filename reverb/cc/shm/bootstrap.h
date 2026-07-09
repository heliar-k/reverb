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

#ifndef REVERB_CC_SHM_BOOTSTRAP_H_
#define REVERB_CC_SHM_BOOTSTRAP_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {

inline constexpr uint32_t kProtocolVersion = 1;  // Hello/Welcome version check

// Server side: a listening Unix domain socket. `Create` unlinks any stale
// socket file first (R7: PID reuse leaves a stale .sock), then binds + listens.
class ShmBootstrapServer {
 public:
  ShmBootstrapServer();
  ~ShmBootstrapServer();
  ShmBootstrapServer(const ShmBootstrapServer&) = delete;
  ShmBootstrapServer& operator=(const ShmBootstrapServer&) = delete;
  ShmBootstrapServer(ShmBootstrapServer&&) noexcept;
  ShmBootstrapServer& operator=(ShmBootstrapServer&&) noexcept;

  // Unlink any stale socket at `socket_path`, then bind + listen.
  static absl::StatusOr<ShmBootstrapServer> Create(
      const std::string& socket_path);

  // Block until a client connects. Returns the connected client fd (caller
  // owns it and must close) and the client's PID (read via SO_PEERCRED).
  absl::StatusOr<std::pair<int, int>> Accept();

  const std::string& socket_path() const { return socket_path_; }

 private:
  int listen_fd_ = -1;
  std::string socket_path_;
};

// Send a WelcomeResponse (length-delimited: 4-byte big-endian length prefix +
// proto bytes) over `client_fd`.
absl::Status SendWelcome(int client_fd, const WelcomeResponse& welcome);

// Receive a HelloRequest (length-delimited) over `client_fd`.
absl::StatusOr<HelloRequest> RecvHello(int client_fd);

// Validate the client's protocol version against kProtocolVersion. Returns
// InvalidArgumentError on mismatch (caller should send an error + close).
absl::Status CheckProtocolVersion(uint32_t client_version);

// Client side: connect to the server's Unix socket and run the Hello/Welcome
// handshake. On success returns the WelcomeResponse (with the three SHM names).
absl::StatusOr<WelcomeResponse> ClientBootstrap(const std::string& socket_path,
                                                int client_pid);

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_BOOTSTRAP_H_
