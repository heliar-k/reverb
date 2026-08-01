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

  // The listening socket fd, for callers (e.g. ShmServer's dispatch loop)
  // that need to poll for new connections without blocking. Spec §8.4: the
  // dispatch loop must not stall on accept while clients have pending work.
  int listen_fd() const { return listen_fd_; }

  const std::string& socket_path() const { return socket_path_; }

 private:
  int listen_fd_ = -1;
  std::string socket_path_;
};

// A3/D-format SHM segment names for one (server, client) pair. The server owns
// generation (spec A3). `server_token` is the server's udsocket path PLUS a
// per-server epoch (ticket #7: PID + boot nanos), sanitized (non-alnum -> '_')
// so the names are valid POSIX shm names. Keying by socket path — NOT by
// server PID alone — is what lets two ShmServers coexist in one process
// (scan #12); the epoch keeps a crash-RESTARTED server on the same socket
// path from colliding with a live client's segments. Clients learn the names
// from Welcome and never need the formula. Decision D splits the control
// rings into a per-flow SPSC pair.
//   /reverb_shm_pool_<token>
//   /reverb_shm_insert_c2s_<token>_<client_pid>
//   /reverb_shm_insert_s2c_<token>_<client_pid>
//   /reverb_shm_sample_c2s_<token>_<client_pid>
//   /reverb_shm_sample_s2c_<token>_<client_pid>
struct ShmSegmentNames {
  std::string pool;
  std::string insert_c2s;
  std::string insert_s2c;
  std::string sample_c2s;
  std::string sample_s2c;
};
ShmSegmentNames MakeShmNames(absl::string_view server_token, int client_pid);

// The server-wide pool segment name for `server_token` (socket path).
std::string MakePoolShmName(absl::string_view server_token);

// Send a WelcomeResponse (length-delimited: 4-byte big-endian length prefix +
// proto bytes) over `client_fd`.
absl::Status SendWelcome(int client_fd, const WelcomeResponse& welcome);

// Receive a HelloRequest (length-delimited) over `client_fd`.
// `timeout` bounds the WHOLE handshake wait (per-byte deadline, airtight
// against a trickling sender). TryAccept on the server passes a small bound —
// it runs on the single dispatch thread, where an unbounded read lets a
// connect-and-stall client wedge accept AND service for everyone.
absl::StatusOr<HelloRequest> RecvHello(
    int client_fd, absl::Duration timeout = absl::InfiniteDuration());

// Validate the client's protocol version against kProtocolVersion. Returns
// InvalidArgumentError on mismatch (caller should send an error + close).
absl::Status CheckProtocolVersion(uint32_t client_version);

// ClientBootstrapResult bundles the handshake response with the open udsocket
// fd. The caller owns the fd and must keep it open for the connection
// lifetime (ticket ⑥): the server `poll()`s this fd for POLLHUP/EOF to detect
// a client crash (spec §8.8). Closing the fd (in ~ShmConnection) is the
// liveness signal.
struct ClientBootstrapResult {
  WelcomeResponse welcome;
  int fd = -1;  // open udsocket fd; caller owns and closes it
};

// Connect, send Hello, recv Welcome, and RETURN the open fd (does not close
// it). Used by ShmClient::Connect, which stores the fd for liveness (ticket ⑥).
absl::StatusOr<ClientBootstrapResult> ClientBootstrapWithFd(
    const std::string& socket_path, int client_pid);

// One-shot handshake: connects, exchanges Hello/Welcome, and CLOSES the fd.
// For callers that do not need a persistent liveness fd (e.g. echo tests).
// ponytail: retained so existing tests/clients are unchanged.
absl::StatusOr<WelcomeResponse> ClientBootstrap(const std::string& socket_path,
                                                int client_pid);

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SHM_BOOTSTRAP_H_
