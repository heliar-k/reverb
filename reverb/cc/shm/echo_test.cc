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

#include <sched.h>
#include <string>
#include <thread>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

// ponytail: poll a non-blocking Read with sched_yield until OK. Ring::Read is
// non-blocking (spec §3.1); the blocking policy is the caller's job (R5). This
// mirrors what the future ShmConnection will do. Duplicated from ring_test
// rather than adding a shared test util — one 8-line helper isn't worth a new
// BUILD target.
absl::Status ReadBlocking(Ring* ring, MsgType* msg_type, std::string* payload) {
  while (true) {
    absl::Status s = ring->Read(msg_type, payload);
    if (s.ok()) return absl::OkStatus();
    if (!absl::IsNotFound(s)) return s;  // real error, surface immediately
    sched_yield();
  }
}

std::string UniqueTag(const std::string& tag) {
  return tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

// End-to-end: data crosses the ring boundary between two threads acting as
// separate processes. Server thread creates the bootstrap socket + both SHM
// rings, accepts the client, sends Welcome. Client thread bootstraps, writes a
// request over C2S, reads the reply over S2C. Server reads the request off
// C2S and writes the reply over S2C.
TEST(ShmEchoTest, DataCrossesRingBoundary) {
  auto tag = UniqueTag("echo");
  std::string sock = "/tmp/reverb_shm_echo_" + tag + ".sock";

  // Bootstrap server.
  auto bs = ShmBootstrapServer::Create(sock);
  REVERB_ASSERT_OK(bs.status());
  ShmBootstrapServer bootstrap = std::move(bs).value();

  const std::string kRequest = "ping-echo-request";
  const std::string kReply = "pong-echo-reply";
  int server_pid = getpid();

  // Server creates the rings AFTER Accept (A3: names need the client PID). The
  // names are server-generated via MakeShmNames, not hardcoded by the test.
  std::thread server_thread([&] {
    auto a = bootstrap.Accept();
    REVERB_ASSERT_OK(a.status());
    auto [client_fd, client_pid] = std::move(a).value();

    auto hello = RecvHello(client_fd);
    REVERB_ASSERT_OK(hello.status());
    REVERB_ASSERT_OK(CheckProtocolVersion(hello->protocol_version()));

    ShmSegmentNames names = MakeShmNames(sock, client_pid);
    auto c2s_server = Ring::Create(names.sample_c2s, 16, 256);
    REVERB_ASSERT_OK(c2s_server.status());
    auto s2c_server = Ring::Create(names.sample_s2c, 16, 256);
    REVERB_ASSERT_OK(s2c_server.status());

    WelcomeResponse welcome;
    welcome.set_pool_shm_name(names.pool);
    welcome.set_sample_c2s_shm_name(names.sample_c2s);
    welcome.set_sample_s2c_shm_name(names.sample_s2c);
    REVERB_ASSERT_OK(SendWelcome(client_fd, welcome));
    close(client_fd);

    // Server reads the request off C2S and writes the reply on S2C.
    MsgType req_type;
    std::string req_payload;
    REVERB_ASSERT_OK(ReadBlocking(&*c2s_server, &req_type, &req_payload));
    EXPECT_EQ(req_type, SAMPLE);
    EXPECT_EQ(req_payload, kRequest);

    REVERB_ASSERT_OK(
        s2c_server->Write(SAMPLE_RESP, absl::MakeSpan(kReply)));
  });

  // Client side: bootstrap, then open the rings by the A3 names from Welcome.
  auto r = ClientBootstrap(sock, /*client_pid=*/getpid());
  REVERB_ASSERT_OK(r.status());
  WelcomeResponse welcome = std::move(r).value();
  // The client asserts it received A3-format names with its own PID.
  int client_pid = getpid();
  ShmSegmentNames expected_names = MakeShmNames(sock, client_pid);
  EXPECT_EQ(welcome.pool_shm_name(), expected_names.pool);
  EXPECT_EQ(welcome.sample_c2s_shm_name(), expected_names.sample_c2s);
  EXPECT_EQ(welcome.sample_s2c_shm_name(), expected_names.sample_s2c);

  auto c2s_client = Ring::Open(welcome.sample_c2s_shm_name());
  REVERB_ASSERT_OK(c2s_client.status());
  auto s2c_client = Ring::Open(welcome.sample_s2c_shm_name());
  REVERB_ASSERT_OK(s2c_client.status());

  // Client writes the request on C2S, reads the reply off S2C.
  REVERB_ASSERT_OK(
      c2s_client->Write(SAMPLE, absl::MakeSpan(kRequest)));
  MsgType reply_type;
  std::string reply_payload;
  REVERB_ASSERT_OK(ReadBlocking(&*s2c_client, &reply_type, &reply_payload));
  EXPECT_EQ(reply_type, SAMPLE_RESP);
  EXPECT_EQ(reply_payload, kReply);

  server_thread.join();
}

// Same as above but the request spans multiple slots, proving cross-slot
// reassembly works across the process boundary.
TEST(ShmEchoTest, CrossSlotMessageAcrossBoundary) {
  auto tag = UniqueTag("crossecho");
  std::string sock = "/tmp/reverb_shm_echo_" + tag + ".sock";

  auto bs = ShmBootstrapServer::Create(sock);
  REVERB_ASSERT_OK(bs.status());
  ShmBootstrapServer bootstrap = std::move(bs).value();

  // 600 bytes -> 3 slots on a 256-byte ring (240-byte body each).
  std::string request(600, 'A');
  std::string reply(900, 'B');  // 4 slots
  int server_pid = getpid();

  std::thread server_thread([&] {
    auto a = bootstrap.Accept();
    REVERB_ASSERT_OK(a.status());
    auto [client_fd, client_pid] = std::move(a).value();
    // A3: server generates the segment names from the real PIDs.
    ShmSegmentNames names = MakeShmNames(sock, client_pid);
    auto c2s_server = Ring::Create(names.sample_c2s, 16, 256);
    REVERB_ASSERT_OK(c2s_server.status());
    auto s2c_server = Ring::Create(names.sample_s2c, 16, 256);
    REVERB_ASSERT_OK(s2c_server.status());
    WelcomeResponse w;
    w.set_pool_shm_name(names.pool);
    w.set_sample_c2s_shm_name(names.sample_c2s);
    w.set_sample_s2c_shm_name(names.sample_s2c);
    REVERB_ASSERT_OK(SendWelcome(client_fd, w));
    close(client_fd);

    MsgType t;
    std::string p;
    REVERB_ASSERT_OK(ReadBlocking(&*c2s_server, &t, &p));
    EXPECT_EQ(p, request);
    REVERB_ASSERT_OK(s2c_server->Write(SAMPLE_RESP, absl::MakeSpan(reply)));
  });

  auto r = ClientBootstrap(sock, getpid());
  REVERB_ASSERT_OK(r.status());
  auto c2s_client = Ring::Open(r->sample_c2s_shm_name());
  REVERB_ASSERT_OK(c2s_client.status());
  auto s2c_client = Ring::Open(r->sample_s2c_shm_name());
  REVERB_ASSERT_OK(s2c_client.status());

  REVERB_ASSERT_OK(c2s_client->Write(SAMPLE, absl::MakeSpan(request)));
  MsgType t;
  std::string p;
  REVERB_ASSERT_OK(ReadBlocking(&*s2c_client, &t, &p));
  EXPECT_EQ(t, SAMPLE_RESP);
  EXPECT_EQ(p, reply);

  server_thread.join();
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
