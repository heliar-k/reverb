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

#include "reverb/cc/shm/bootstrap.h"

#include <cerrno>
#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/platform/default/status_macros.h"
#include <arpa/inet.h>  // htonl/ntohl
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace deepmind {
namespace reverb {
namespace shm {

namespace {

// Read exactly `n` bytes from fd (loop over partial reads). Returns
// InvalidArgumentError on EOF (peer closed) or InternalError on read error.
absl::Status ReadExact(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  size_t got = 0;
  while (got < n) {
    ssize_t r = read(fd, p + got, n - got);
    if (r < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          absl::StrCat("read failed (errno ", errno, ": ",
                       std::strerror(errno), ")"));
    }
    if (r == 0) {
      return absl::InvalidArgumentError("connection closed by peer");
    }
    got += r;
  }
  return absl::OkStatus();
}

absl::Status WriteAll(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  size_t sent = 0;
  while (sent < n) {
    ssize_t w = write(fd, p + sent, n - sent);
    if (w < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          absl::StrCat("write failed (errno ", errno, ": ",
                       std::strerror(errno), ")"));
    }
    sent += w;
  }
  return absl::OkStatus();
}

absl::Status ErrnoStatus(std::string_view op, std::string_view detail) {
  return absl::InternalError(
      absl::StrCat(op, " failed: ", detail, " (errno ", errno, ": ",
                   std::strerror(errno), ")"));
}

}  // namespace

ShmBootstrapServer::ShmBootstrapServer() = default;

ShmBootstrapServer::~ShmBootstrapServer() {
  if (listen_fd_ >= 0) close(listen_fd_);
  if (!socket_path_.empty()) unlink(socket_path_.c_str());
}

ShmBootstrapServer::ShmBootstrapServer(ShmBootstrapServer&& other) noexcept
    : listen_fd_(other.listen_fd_), socket_path_(std::move(other.socket_path_)) {
  other.listen_fd_ = -1;
  other.socket_path_.clear();
}

ShmBootstrapServer& ShmBootstrapServer::operator=(
    ShmBootstrapServer&& other) noexcept {
  if (this != &other) {
    if (listen_fd_ >= 0) close(listen_fd_);
    if (!socket_path_.empty()) unlink(socket_path_.c_str());
    listen_fd_ = other.listen_fd_;
    socket_path_ = std::move(other.socket_path_);
    other.listen_fd_ = -1;
    other.socket_path_.clear();
  }
  return *this;
}

// static
absl::StatusOr<ShmBootstrapServer> ShmBootstrapServer::Create(
    const std::string& socket_path) {
  // R7: unlink a stale socket left by a crashed previous server before bind.
  unlink(socket_path.c_str());

  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return ErrnoStatus("socket", socket_path);

  struct sockaddr_un addr {};
  addr.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(addr.sun_path)) {
    close(fd);
    return absl::InvalidArgumentError("socket_path too long");
  }
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    auto s = ErrnoStatus("bind", socket_path);
    close(fd);
    return s;
  }
  if (listen(fd, /*backlog=*/5) < 0) {
    auto s = ErrnoStatus("listen", socket_path);
    close(fd);
    unlink(socket_path.c_str());
    return s;
  }

  ShmBootstrapServer server;
  server.listen_fd_ = fd;
  server.socket_path_ = socket_path;
  return server;
}

absl::StatusOr<std::pair<int, int>> ShmBootstrapServer::Accept() {
  int fd = accept(listen_fd_, nullptr, nullptr);
  if (fd < 0) {
    if (errno == EINTR) {
      return absl::CancelledError("interrupted");
    }
    return ErrnoStatus("accept", socket_path_);
  }
  // Read the peer PID via SO_PEERCRED (Linux).
  struct ucred cred {};
  socklen_t len = sizeof(cred);
  int pid = -1;
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) == 0) {
    pid = cred.pid;
  }
  return std::make_pair(fd, pid);
}

absl::Status SendWelcome(int client_fd, const WelcomeResponse& welcome) {
  std::string body;
  if (!welcome.SerializeToString(&body)) {
    return absl::InternalError("failed to serialize WelcomeResponse");
  }
  uint32_t len = htonl(static_cast<uint32_t>(body.size()));
  REVERB_RETURN_IF_ERROR(WriteAll(client_fd, &len, sizeof(len)));
  REVERB_RETURN_IF_ERROR(WriteAll(client_fd, body.data(), body.size()));
  return absl::OkStatus();
}

// Poll until readable or past `deadline` (EINTR-safe).
absl::Status PollReadable(int fd, absl::Time deadline) {
  while (true) {
    int64_t ms = absl::ToInt64Milliseconds(deadline - absl::Now());
    if (ms < 0) ms = 0;
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int r = poll(&pfd, 1, static_cast<int>(ms));
    if (r > 0) return absl::OkStatus();
    if (r == 0) return absl::DeadlineExceededError("recv timeout");
    if (errno == EINTR) continue;
    return ErrnoStatus("poll", "recv");
  }
}

// ReadExact with a hard deadline: polls before EVERY read, so a peer that
// trickles partial bytes then stalls cannot block past the deadline.
absl::Status ReadExactBounded(int fd, void* buf, size_t n,
                              absl::Time deadline) {
  char* p = static_cast<char*>(buf);
  size_t got = 0;
  while (got < n) {
    REVERB_RETURN_IF_ERROR(PollReadable(fd, deadline));
    ssize_t r = read(fd, p + got, n - got);
    if (r < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          absl::StrCat("read failed (errno ", errno, ": ",
                       std::strerror(errno), ")"));
    }
    if (r == 0) {
      return absl::InvalidArgumentError("connection closed by peer");
    }
    got += r;
  }
  return absl::OkStatus();
}

absl::StatusOr<HelloRequest> RecvHello(int client_fd, absl::Duration timeout) {
  const bool bounded = timeout != absl::InfiniteDuration();
  const absl::Time deadline = absl::Now() + timeout;
  auto read = [&](void* buf, size_t n) {
    return bounded ? ReadExactBounded(client_fd, buf, n, deadline)
                   : ReadExact(client_fd, buf, n);
  };
  uint32_t len_net = 0;
  REVERB_RETURN_IF_ERROR(read(&len_net, sizeof(len_net)));
  uint32_t len = ntohl(len_net);
  // ponytail: cap at 4MB to reject a hostile/huge length prefix; Hello is tiny.
  if (len > 4 * 1024 * 1024) {
    return absl::InvalidArgumentError("HelloRequest length too large");
  }
  std::string body(len, '\0');
  REVERB_RETURN_IF_ERROR(read(body.data(), len));
  HelloRequest hello;
  if (!hello.ParseFromString(body)) {
    return absl::InvalidArgumentError("failed to parse HelloRequest");
  }
  return hello;
}

absl::Status CheckProtocolVersion(uint32_t client_version) {
  if (client_version != kProtocolVersion) {
    return absl::InvalidArgumentError(
        absl::StrCat("protocol version mismatch: client=", client_version,
                     " server=", kProtocolVersion));
  }
  return absl::OkStatus();
}

namespace {
// POSIX shm names: single leading '/', no further slashes. Map the server's
// socket path to a valid, filesystem-unique component.
std::string SanitizeToken(absl::string_view token) {
  std::string out;
  out.reserve(token.size());
  for (char c : token) {
    out.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  }
  return out;
}
}  // namespace

std::string MakePoolShmName(absl::string_view server_token) {
  return absl::StrCat("/reverb_shm_pool_", SanitizeToken(server_token));
}

ShmSegmentNames MakeShmNames(absl::string_view server_token, int client_pid) {
  const std::string tok = SanitizeToken(server_token);
  ShmSegmentNames names;
  names.pool = MakePoolShmName(server_token);
  names.insert_c2s =
      absl::StrCat("/reverb_shm_insert_c2s_", tok, "_", client_pid);
  names.insert_s2c =
      absl::StrCat("/reverb_shm_insert_s2c_", tok, "_", client_pid);
  names.sample_c2s =
      absl::StrCat("/reverb_shm_sample_c2s_", tok, "_", client_pid);
  names.sample_s2c =
      absl::StrCat("/reverb_shm_sample_s2c_", tok, "_", client_pid);
  return names;
}

absl::StatusOr<ClientBootstrapResult> ClientBootstrapWithFd(
    const std::string& socket_path, int client_pid) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return ErrnoStatus("socket", socket_path);

  struct sockaddr_un addr {};
  addr.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(addr.sun_path)) {
    close(fd);
    return absl::InvalidArgumentError("socket_path too long");
  }
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    auto s = ErrnoStatus("connect", socket_path);
    close(fd);
    return s;
  }

  HelloRequest hello;
  hello.set_client_pid(client_pid);
  hello.set_protocol_version(kProtocolVersion);
  std::string body;
  hello.SerializeToString(&body);
  uint32_t len = htonl(static_cast<uint32_t>(body.size()));
  auto write_status = WriteAll(fd, &len, sizeof(len));
  if (!write_status.ok()) {
    close(fd);
    return write_status;
  }
  auto send_status = WriteAll(fd, body.data(), body.size());
  if (!send_status.ok()) {
    close(fd);
    return send_status;
  }

  // Read the WelcomeResponse.
  uint32_t resp_len_net = 0;
  auto read_len = ReadExact(fd, &resp_len_net, sizeof(resp_len_net));
  if (!read_len.ok()) {
    close(fd);
    return read_len;
  }
  uint32_t resp_len = ntohl(resp_len_net);
  if (resp_len > 64 * 1024 * 1024) {
    close(fd);
    return absl::InvalidArgumentError("WelcomeResponse length too large");
  }
  std::string resp_body(resp_len, '\0');
  auto read_body = ReadExact(fd, resp_body.data(), resp_len);
  if (!read_body.ok()) {
    close(fd);
    return read_body;
  }
  // NOTE: do NOT close `fd` — the caller owns it and keeps it open for the
  // connection lifetime as the liveness signal (ticket ⑥, spec §8.8).

  ClientBootstrapResult result;
  if (!result.welcome.ParseFromString(resp_body)) {
    close(fd);
    return absl::InternalError("failed to parse WelcomeResponse");
  }
  result.fd = fd;
  return result;
}

absl::StatusOr<WelcomeResponse> ClientBootstrap(const std::string& socket_path,
                                                int client_pid) {
  auto r = ClientBootstrapWithFd(socket_path, client_pid);
  if (!r.ok()) return r.status();
  // One-shot callers don't need the liveness fd: close it and return just the
  // Welcome. ponytail: retained so existing echo/bootstrap tests are unchanged.
  close(r->fd);
  return std::move(r->welcome);
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
