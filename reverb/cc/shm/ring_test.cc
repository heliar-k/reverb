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

#include <string>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

using ::testing::HasSubstr;

// A unique SHM name per test to avoid collisions across parallel runs.
std::string UniqueName(const std::string& tag) {
  return "/reverb_shm_ring_test_" + tag + "_" +
         std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

TEST(RingTest, SingleSlotWriteRead) {
  auto s = Ring::Create(UniqueName("single"), 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  std::string payload = "hello";
  REVERB_ASSERT_OK(ring.Write(HELLO, absl::MakeSpan(payload)));
  MsgType type;
  std::string out;
  REVERB_ASSERT_OK(ring.Read(&type, &out));
  EXPECT_EQ(type, HELLO);
  EXPECT_EQ(out, payload);
}

TEST(RingTest, CrossSlotReassembly) {
  // slot_size 256, SlotHeader 16 -> 240 bytes/slot body. A 600-byte payload
  // spans 3 slots.
  auto s = Ring::Create(UniqueName("cross"), 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  std::string payload(600, 'x');
  for (size_t i = 0; i < payload.size(); i++) payload[i] = static_cast<char>(i);
  REVERB_ASSERT_OK(ring.Write(INSERT, absl::MakeSpan(payload)));
  MsgType type;
  std::string out;
  REVERB_ASSERT_OK(ring.Read(&type, &out));
  EXPECT_EQ(type, INSERT);
  EXPECT_EQ(out, payload);
}

TEST(RingTest, FullBlocksThenUnblocks) {
  auto s = Ring::Create(UniqueName("full"), 4, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  // Fill all 4 slots with 4 separate single-slot messages.
  for (int i = 0; i < 4; i++) {
    std::string p = "m" + std::to_string(i);
    REVERB_ASSERT_OK(ring.Write(SAMPLE, absl::MakeSpan(p)));
  }
  // A 5th write must block (ring full). Run it on a thread; drain the ring so
  // the write completes.
  std::thread writer([&] {
    std::string p = "unblock";
    REVERB_ASSERT_OK(ring.Write(RELEASE, absl::MakeSpan(p)));
  });
  // Drain: read until we see the "unblock" message. Each read frees a slot.
  MsgType type;
  std::string out;
  bool saw_unblock = false;
  for (int i = 0; i < 5; i++) {
    REVERB_ASSERT_OK(ring.Read(&type, &out));
    if (out == "unblock") saw_unblock = true;
  }
  writer.join();
  EXPECT_TRUE(saw_unblock);
}

TEST(RingTest, EmptyBlocksThenUnblocks) {
  auto s = Ring::Create(UniqueName("empty"), 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  std::thread reader([&] {
    MsgType type;
    std::string out;
    REVERB_ASSERT_OK(ring.Read(&type, &out));
    EXPECT_EQ(type, WELCOME);
    EXPECT_EQ(out, "ready");
  });
  // Give the reader a chance to block on the empty ring.
  std::string p = "ready";
  REVERB_ASSERT_OK(ring.Write(WELCOME, absl::MakeSpan(p)));
  reader.join();
}

TEST(RingTest, MultiRoundWraparound) {
  // capacity 4: write/read 100 messages to exercise seq wrap past capacity
  // multiple times and slot reuse.
  auto s = Ring::Create(UniqueName("wrap"), 4, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  for (int i = 0; i < 100; i++) {
    std::string p = "msg-" + std::to_string(i);
    REVERB_ASSERT_OK(ring.Write(HELLO, absl::MakeSpan(p)));
    MsgType type;
    std::string out;
    REVERB_ASSERT_OK(ring.Read(&type, &out));
    EXPECT_EQ(type, HELLO);
    EXPECT_EQ(out, p);
  }
}

TEST(RingTest, SeqCounterWraps) {
  // seq is uint64 starting at 1; we can't overflow it in a test, but we verify
  // the ring keeps working after many rounds (head/tail advance far beyond
  // capacity) which is the wraparound correctness property.
  auto s = Ring::Create(UniqueName("seqwrap"), 4, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  for (int i = 0; i < 1000; i++) {
    std::string p(1, static_cast<char>(i & 0xff));
    REVERB_ASSERT_OK(ring.Write(HELLO, absl::MakeSpan(p)));
    MsgType type;
    std::string out;
    REVERB_ASSERT_OK(ring.Read(&type, &out));
    ASSERT_EQ(out, p);
  }
}

TEST(RingTest, MessageTooLargeRejected) {
  auto s = Ring::Create(UniqueName("toolarge"), 4, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  // 240 bytes/slot * 4 slots = 960 max; 2000 bytes exceeds capacity.
  std::string payload(2000, 'z');
  absl::Status st = ring.Write(INSERT, absl::MakeSpan(payload));
  EXPECT_FALSE(st.ok());
  EXPECT_THAT(std::string(st.message()), HasSubstr("capacity"));
}

TEST(RingTest, TwoRingsSameSegment) {
  // The owner and an opener see the same data: simulate two sides by creating
  // one Ring and opening another against the same segment.
  auto name = UniqueName("twoside");
  auto ss = Ring::Create(name, 16, 256);
  REVERB_ASSERT_OK(ss.status());
  Ring server = std::move(ss).value();
  auto sc = Ring::Open(name);
  REVERB_ASSERT_OK(sc.status());
  Ring client = std::move(sc).value();
  std::string payload = "across-the-ring";
  REVERB_ASSERT_OK(server.Write(SAMPLE, absl::MakeSpan(payload)));
  MsgType type;
  std::string out;
  REVERB_ASSERT_OK(client.Read(&type, &out));
  EXPECT_EQ(type, SAMPLE);
  EXPECT_EQ(out, payload);
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
