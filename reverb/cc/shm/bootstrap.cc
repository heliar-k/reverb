#include "reverb/cc/shm/bootstrap.h"

#include <cerrno>
#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/platform/status_macros.h"
#include <arpa/inet.h>  // htonl/ntohl
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

absl::StatusOr<HelloRequest> RecvHello(int client_fd) {
  uint32_t len_net = 0;
  REVERB_RETURN_IF_ERROR(ReadExact(client_fd, &len_net, sizeof(len_net)));
  uint32_t len = ntohl(len_net);
  // ponytail: cap at 4MB to reject a hostile/huge length prefix; Hello is tiny.
  if (len > 4 * 1024 * 1024) {
    return absl::InvalidArgumentError("HelloRequest length too large");
  }
  std::string body(len, '\0');
  REVERB_RETURN_IF_ERROR(ReadExact(client_fd, body.data(), len));
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

absl::StatusOr<WelcomeResponse> ClientBootstrap(const std::string& socket_path,
                                                int client_pid) {
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
  REVERB_RETURN_IF_ERROR(WriteAll(fd, &len, sizeof(len)));
  REVERB_RETURN_IF_ERROR(WriteAll(fd, body.data(), body.size()));

  // Read the WelcomeResponse.
  uint32_t resp_len_net = 0;
  REVERB_RETURN_IF_ERROR(ReadExact(fd, &resp_len_net, sizeof(resp_len_net)));
  uint32_t resp_len = ntohl(resp_len_net);
  if (resp_len > 64 * 1024 * 1024) {
    close(fd);
    return absl::InvalidArgumentError("WelcomeResponse length too large");
  }
  std::string resp_body(resp_len, '\0');
  REVERB_RETURN_IF_ERROR(ReadExact(fd, resp_body.data(), resp_len));
  close(fd);

  WelcomeResponse welcome;
  if (!welcome.ParseFromString(resp_body)) {
    return absl::InternalError("failed to parse WelcomeResponse");
  }
  return welcome;
}

}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
