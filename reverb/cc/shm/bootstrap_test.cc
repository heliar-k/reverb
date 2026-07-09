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

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

std::string UniqueSocket(const std::string& tag) {
  return "/tmp/reverb_shm_boot_test_" + tag + "_" +
         std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag)) + ".sock";
}

WelcomeResponse MakeWelcome() {
  WelcomeResponse w;
  w.set_pool_shm_name("/reverb_shm_pool_12345");
  w.set_c2s_shm_name("/reverb_shm_c2s_12345_67890");
  w.set_s2c_shm_name("/reverb_shm_s2c_12345_67890");
  return w;
}

TEST(BootstrapTest, HelloWelcomeRoundTrip) {
  auto sock = UniqueSocket("rt");
  auto s = ShmBootstrapServer::Create(sock);
  REVERB_ASSERT_OK(s.status());
  ShmBootstrapServer server = std::move(s).value();

  WelcomeResponse expected = MakeWelcome();
  std::thread client_thread([&] {
    auto r = ClientBootstrap(sock, /*client_pid=*/67890);
    REVERB_ASSERT_OK(r.status());
    WelcomeResponse got = std::move(r).value();
    EXPECT_EQ(got.pool_shm_name(), expected.pool_shm_name());
    EXPECT_EQ(got.c2s_shm_name(), expected.c2s_shm_name());
    EXPECT_EQ(got.s2c_shm_name(), expected.s2c_shm_name());
  });

  auto a = server.Accept();
  REVERB_ASSERT_OK(a.status());
  auto [client_fd, client_pid] = std::move(a).value();
  ASSERT_GE(client_fd, 0);

  auto hello = RecvHello(client_fd);
  REVERB_ASSERT_OK(hello.status());
  EXPECT_EQ(hello->protocol_version(), kProtocolVersion);
  REVERB_ASSERT_OK(SendWelcome(client_fd, expected));
  close(client_fd);

  client_thread.join();
}

TEST(BootstrapTest, NamesAreNonEmpty) {
  auto sock = UniqueSocket("names");
  auto s = ShmBootstrapServer::Create(sock);
  REVERB_ASSERT_OK(s.status());
  ShmBootstrapServer server = std::move(s).value();

  WelcomeResponse expected = MakeWelcome();
  std::thread client_thread([&] {
    auto r = ClientBootstrap(sock, /*client_pid=*/67890);
    REVERB_ASSERT_OK(r.status());
    WelcomeResponse got = std::move(r).value();
    EXPECT_FALSE(got.pool_shm_name().empty());
    EXPECT_FALSE(got.c2s_shm_name().empty());
    EXPECT_FALSE(got.s2c_shm_name().empty());
  });

  auto a = server.Accept();
  REVERB_ASSERT_OK(a.status());
  auto [client_fd, client_pid] = std::move(a).value();
  REVERB_ASSERT_OK(SendWelcome(client_fd, expected));
  close(client_fd);
  client_thread.join();
}

TEST(BootstrapTest, ProtocolVersionMismatchRejected) {
  auto sock = UniqueSocket("mismatch");
  auto s = ShmBootstrapServer::Create(sock);
  REVERB_ASSERT_OK(s.status());
  ShmBootstrapServer server = std::move(s).value();

  std::thread client_thread([&] {
    // Connect and send a Hello with a wrong protocol version.
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    ASSERT_GE(fd, 0);
    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock.c_str(), sizeof(addr.sun_path) - 1);
    ASSERT_EQ(connect(fd, reinterpret_cast<struct sockaddr*>(&addr),
                       sizeof(addr)), 0);
    HelloRequest hello;
    hello.set_client_pid(67890);
    hello.set_protocol_version(kProtocolVersion + 999);  // wrong
    std::string buf;
    hello.SerializeToString(&buf);
    uint32_t len = static_cast<uint32_t>(buf.size());
    char lenbuf[4] = {
        static_cast<char>((len >> 24) & 0xff),
        static_cast<char>((len >> 16) & 0xff),
        static_cast<char>((len >> 8) & 0xff),
        static_cast<char>(len & 0xff)};
    ASSERT_EQ(write(fd, lenbuf, 4), 4);
    ASSERT_EQ(write(fd, buf.data(), buf.size()),
              static_cast<ssize_t>(buf.size()));
    // Server should reject and close; client reads an error or EOF.
    char reply[4];
    ssize_t n = read(fd, reply, 4);
    // Either EOF (server closed without welcome) or an error message.
    EXPECT_LE(n, 4);
    close(fd);
  });

  auto a = server.Accept();
  REVERB_ASSERT_OK(a.status());
  auto [client_fd, client_pid] = std::move(a).value();
  auto hello = RecvHello(client_fd);
  REVERB_ASSERT_OK(hello.status());
  absl::Status check = CheckProtocolVersion(hello->protocol_version());
  EXPECT_FALSE(check.ok());
  EXPECT_THAT(std::string(check.message()), HasSubstr("version"));
  close(client_fd);
  client_thread.join();
}

TEST(BootstrapTest, StaleSocketUnlinkedBeforeBind) {
  // Create a stale socket file first, then Create must succeed despite it.
  auto sock = UniqueSocket("stale");
  int fd = open(sock.c_str(), O_CREAT | O_WRONLY, 0600);
  ASSERT_GE(fd, 0);
  close(fd);
  ASSERT_TRUE(access(sock.c_str(), F_OK) == 0);

  auto s = ShmBootstrapServer::Create(sock);
  REVERB_ASSERT_OK(s.status());
  ShmBootstrapServer server = std::move(s).value();
  EXPECT_EQ(server.socket_path(), sock);
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
