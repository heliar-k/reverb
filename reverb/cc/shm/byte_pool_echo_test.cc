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

// End-to-end proof of the C4 allocation-ownership model: the server is the
// sole pool allocator; a client asks for an offset over the C→S ring, gets it
// back over the S→C ring, memcpy's bytes into the granted region, then sends
// RELEASE so the server can reclaim it. Reuses ①'s Ring + bootstrap and ②'s
// ShmBytePool.

#include <sched.h>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/byte_pool.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

// ponytail: poll a non-blocking Read with sched_yield until OK. Ring::Read is
// non-blocking (spec §3.1); the blocking policy is the caller's job (R5).
// Duplicated from echo_test rather than adding a shared test util.
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

// The slab geometry the server uses for this test.
const size_t kSlabs[] = {64, 256, 1024, 4096};
const size_t kBlocksPerSlab = 4;

TEST(ShmBytePoolEchoTest, ClientAllocatesViaRingThenReleases) {
  auto tag = UniqueTag("bpecho");
  std::string sock = "/tmp/reverb_shm_bpecho_" + tag + ".sock";
  int server_pid = getpid();

  // The payload the client will memcpy into the granted region. 200 bytes fits
  // the 256-byte tier (and is too big for the 64-byte tier), exercising slab
  // selection across the ring.
  const std::string kPayload(200, 'Z');
  const uint64_t kNumBytes = kPayload.size();

  // Server side: bootstrap + pool + rings. The dispatch loop handles one
  // ALLOCATE then one RELEASE, then stops.
  std::thread server_thread([&] {
    // Bootstrap: accept the client and learn its PID (needed for ring names).
    auto bs = ShmBootstrapServer::Create(sock);
    REVERB_ASSERT_OK(bs.status());
    ShmBootstrapServer bootstrap = std::move(bs).value();

    auto a = bootstrap.Accept();
    REVERB_ASSERT_OK(a.status());
    auto [client_fd, client_pid] = std::move(a).value();
    auto hello = RecvHello(client_fd);
    REVERB_ASSERT_OK(hello.status());
    REVERB_ASSERT_OK(CheckProtocolVersion(hello->protocol_version()));

    ShmSegmentNames names = MakeShmNames(server_pid, client_pid);

    // Create the pool + both rings. The pool name is the A3 pool name.
    auto pool_s = ShmBytePool::Create(names.pool, kSlabs, kBlocksPerSlab);
    REVERB_ASSERT_OK(pool_s.status());
    ShmBytePool pool = std::move(pool_s).value();
    auto c2s = Ring::Create(names.insert_c2s, 16, 256);
    REVERB_ASSERT_OK(c2s.status());
    auto s2c = Ring::Create(names.insert_s2c, 16, 256);
    REVERB_ASSERT_OK(s2c.status());

    WelcomeResponse welcome;
    welcome.set_pool_shm_name(names.pool);
    welcome.set_insert_c2s_shm_name(names.insert_c2s);
    welcome.set_insert_s2c_shm_name(names.insert_s2c);
    REVERB_ASSERT_OK(SendWelcome(client_fd, welcome));
    close(client_fd);

    // ---- Dispatch one ALLOCATE ----
    MsgType req_type;
    std::string req_payload;
    REVERB_ASSERT_OK(ReadBlocking(&*c2s, &req_type, &req_payload));
    EXPECT_EQ(req_type, ALLOCATE);
    ShmAllocateRequest alloc_req;
    ASSERT_TRUE(alloc_req.ParseFromString(req_payload));
    EXPECT_EQ(alloc_req.num_bytes(), kNumBytes);

    // Server is the sole allocator (C4): allocate, then hand the offset back.
    auto off = pool.Allocate(alloc_req.num_bytes());
    REVERB_ASSERT_OK(off.status());
    // C3: server sets refcount=1 when it grants a block it expects the client
    // to release (here the client writes/owns it, so track it as outstanding).
    pool.Ref(*off);

    ShmAllocateResponse alloc_resp;
    alloc_resp.set_shm_offset(*off);
    std::string resp_body;
    alloc_resp.SerializeToString(&resp_body);
    REVERB_ASSERT_OK(s2c->Write(ALLOCATE_RESP, absl::MakeSpan(resp_body)));

    // ---- Dispatch one RELEASE ----
    MsgType rel_type;
    std::string rel_payload;
    REVERB_ASSERT_OK(ReadBlocking(&*c2s, &rel_type, &rel_payload));
    EXPECT_EQ(rel_type, RELEASE);
    ShmReleaseRequest rel_req;
    ASSERT_TRUE(rel_req.ParseFromString(rel_payload));
    ASSERT_EQ(rel_req.offsets_size(), 1);
    EXPECT_EQ(rel_req.offsets(0), *off);

    // Server reclaims: Unref -> 0 means Deallocate (C3).
    EXPECT_TRUE(pool.Unref(rel_req.offsets(0)));
    pool.Deallocate(rel_req.offsets(0));

    // The just-freed block should recycle on the next same-size allocate.
    auto off2 = pool.Allocate(kNumBytes);
    REVERB_ASSERT_OK(off2.status());
    EXPECT_EQ(*off2, *off);
  });

  // Client side: bootstrap, open pool + rings, run the C4 round-trip. The
  // server thread may not have bound the socket yet, so retry connect for a
  // short window (a real client does the same).
  WelcomeResponse welcome;
  {
    absl::StatusOr<WelcomeResponse> r;
    for (int i = 0; i < 200; i++) {
      r = ClientBootstrap(sock, /*client_pid=*/getpid());
      if (r.ok()) break;
      sched_yield();
    }
    REVERB_ASSERT_OK(r.status()) << r.status();
    welcome = std::move(r).value();
  }

  auto pool_co = ShmBytePool::Open(welcome.pool_shm_name());
  REVERB_ASSERT_OK(pool_co.status());
  ShmBytePool client_pool = std::move(pool_co).value();
  auto c2s = Ring::Open(welcome.insert_c2s_shm_name());
  REVERB_ASSERT_OK(c2s.status());
  auto s2c = Ring::Open(welcome.insert_s2c_shm_name());
  REVERB_ASSERT_OK(s2c.status());

  // 1. Ask the server for an offset (C4: client never self-allocates).
  ShmAllocateRequest alloc_req;
  alloc_req.set_num_bytes(kNumBytes);
  std::string req_body;
  alloc_req.SerializeToString(&req_body);
  REVERB_ASSERT_OK(c2s->Write(ALLOCATE, absl::MakeSpan(req_body)));

  // 2. Read the granted offset back.
  MsgType resp_type;
  std::string resp_body;
  REVERB_ASSERT_OK(ReadBlocking(&*s2c, &resp_type, &resp_body));
  EXPECT_EQ(resp_type, ALLOCATE_RESP);
  ShmAllocateResponse alloc_resp;
  ASSERT_TRUE(alloc_resp.ParseFromString(resp_body));
  uint64_t offset = alloc_resp.shm_offset();

  // 3. Client writes bytes into the granted region (RW mmap per C4).
  std::memcpy(client_pool.At(offset), kPayload.data(), kPayload.size());

  // 4. Client reads its own bytes back through its mapping to confirm.
  std::string readback(kPayload.size(), '\0');
  std::memcpy(&readback[0], client_pool.At(offset), kPayload.size());
  EXPECT_EQ(readback, kPayload);

  // 5. RELEASE the offset so the server can reclaim it (C3).
  ShmReleaseRequest rel_req;
  rel_req.add_offsets(offset);
  std::string rel_body;
  rel_req.SerializeToString(&rel_body);
  REVERB_ASSERT_OK(c2s->Write(RELEASE, absl::MakeSpan(rel_body)));

  server_thread.join();
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
