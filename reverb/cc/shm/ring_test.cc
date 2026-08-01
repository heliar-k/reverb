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

#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <functional>
#include <sched.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/shm/ring.h"
#include "reverb/cc/shm/shm_connection.h"
#include "reverb/cc/shm/shm_protocol.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

using ::testing::HasSubstr;

// ponytail: poll a non-blocking Read with sched_yield until it returns OK. The
// blocking policy lives at the caller layer (spec R5), never on Ring, so tests
// that need blocking semantics spin here instead of asking Ring to block.
absl::Status ReadBlocking(Ring* ring, MsgType* msg_type, std::string* payload) {
  while (true) {
    absl::Status s = ring->Read(msg_type, payload);
    if (s.ok()) return absl::OkStatus();
    if (!absl::IsNotFound(s)) return s;  // real error, surface immediately
    sched_yield();
  }
}

// A unique SHM name per test to avoid collisions across parallel runs.
std::string UniqueName(const std::string& tag) {
  return "/reverb_shm_ring_test_" + tag + "_" +
         std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

// Map the existing segment `name` and run `fn` on its raw bytes, to simulate
// a corrupted/stale segment (ticket #7: Open/Read defensive validation).
void PokeSegment(const std::string& name, size_t len,
                 std::function<void(char*)> fn) {
  int fd = shm_open(name.c_str(), O_RDWR, 0600);
  ASSERT_GE(fd, 0);
  void* base = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  ASSERT_NE(base, MAP_FAILED);
  fn(static_cast<char*>(base));
  munmap(base, len);
}

// Fills `ring` with 1-slot messages until TryWrite reports RING_FULL, so a
// subsequent WriteBlocking has to WAIT (the only situation where its
// liveness/deadline checks run).
void FillRing(Ring* ring) {
  std::string payload = "x";
  while (true) {
    absl::Status s = ring->TryWrite(SAMPLE, absl::MakeSpan(payload));
    if (absl::IsResourceExhausted(s)) return;
    REVERB_ASSERT_OK(s);
  }
}

// review #2: WriteBlocking must fail fast when the peer dies mid-wait,
// instead of spinning on sched_yield forever (the old Ring::Write behaviour
// — 100% CPU hang with a wedged/dead server). Deterministic EOF: a
// socketpair plays the control_fd role, closing the peer end is the
// liveness signal.
TEST(RingTest, WriteBlockingFailsFastOnPeerDeath) {
  auto s = Ring::Create(UniqueName("wb_eof"), 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  FillRing(&ring);

  int fds[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  std::string payload = "y";
  std::atomic<absl::StatusCode> result{absl::StatusCode::kOk};
  std::thread writer([&] {
    result.store(
        WriteBlocking(&ring, SAMPLE, absl::MakeSpan(payload), fds[0],
                      absl::Seconds(30))
            .code());
  });
  // Let the writer settle into the wait loop (ring is full).
  absl::SleepFor(absl::Milliseconds(100));
  close(fds[1]);  // peer death -> IsPeerClosed(fds[0]) sees EOF
  writer.join();
  close(fds[0]);
  EXPECT_EQ(result.load(), absl::StatusCode::kUnavailable);
}

// review #2: with no liveness signal (control_fd=-1) and a peer that never
// reads, WriteBlocking must still give up at the deadline instead of
// spinning forever.
TEST(RingTest, WriteBlockingRespectsDeadline) {
  auto s = Ring::Create(UniqueName("wb_cap"), 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  FillRing(&ring);

  std::string payload = "y";
  absl::Time start = absl::Now();
  absl::Status st = WriteBlocking(&ring, SAMPLE, absl::MakeSpan(payload),
                                  /*control_fd=*/-1,
                                  absl::Milliseconds(200));
  EXPECT_EQ(st.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_LT(absl::Now() - start, absl::Seconds(5))
      << "WriteBlocking overshot its 200ms deadline";
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
  // The 5th message is written by `writer` only after a slot frees, so its read
  // may transiently return NOT_READY (now that Read is non-blocking) — poll.
  MsgType type;
  std::string out;
  bool saw_unblock = false;
  for (int i = 0; i < 5; i++) {
    REVERB_ASSERT_OK(ReadBlocking(&ring, &type, &out));
    if (out == "unblock") saw_unblock = true;
  }
  writer.join();
  EXPECT_TRUE(saw_unblock);
}

TEST(RingTest, EmptyReadsNonBlockingThenDataArrives) {
  auto s = Ring::Create(UniqueName("empty"), 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  // Read on an empty ring returns NOT_READY immediately (no blocking).
  MsgType type;
  std::string out;
  absl::Status st = ring.Read(&type, &out);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::IsNotFound(st)) << st;
  EXPECT_THAT(std::string(st.message()), HasSubstr("NOT_READY"));

  // A reader thread polls until data arrives (R5: blocking is the caller's job).
  std::thread reader([&] {
    MsgType t;
    std::string o;
    REVERB_ASSERT_OK(ReadBlocking(&ring, &t, &o));
    EXPECT_EQ(t, WELCOME);
    EXPECT_EQ(o, "ready");
  });
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

TEST(RingTest, MultiSlotConcurrentReadNeverSeesPartialMessage) {
  // Regression for the multi-slot publish race: WriteSlots used to release
  // each slot's seq in order 0..n-1, so a consumer that passed slot 0 could
  // find slot k+1 still unpublished (producer preempted mid-loop) and get
  // InternalError("ring continuation slot missing") on a NORMAL race —
  // fatal to the client's ReadBlocking. The fix publishes slot 0's seq LAST,
  // so the consumer's acquire on slot 0 makes the whole message visible
  // atomically. Every Read must be NOT_READY or a complete intact message.
  auto s = Ring::Create(UniqueName("race"), 128, 64);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();

  constexpr int kMessages = 500;
  constexpr size_t kPayloadLen = 4800;  // 100 slots x 48-byte body
  std::atomic<bool> reader_failed{false};
  std::atomic<int> read_count{0};
  absl::Status reader_status = absl::OkStatus();  // read after join

  std::thread consumer([&] {
    MsgType type;
    std::string out;
    while (read_count.load() < kMessages && !reader_failed.load()) {
      absl::Status st = ring.Read(&type, &out);
      if (absl::IsNotFound(st)) {
        sched_yield();
        continue;
      }
      if (!st.ok()) {
        reader_status = st;
        reader_failed.store(true);
        return;
      }
      int idx = read_count.load();
      EXPECT_EQ(type, INSERT);
      EXPECT_EQ(out, std::string(kPayloadLen, static_cast<char>(idx % 251)))
          << "corrupted payload at message " << idx;
      read_count.store(idx + 1);
    }
  });

  std::thread producer([&] {
    for (int i = 0; i < kMessages && !reader_failed.load(); i++) {
      std::string payload(kPayloadLen, static_cast<char>(i % 251));
      REVERB_ASSERT_OK(ring.Write(INSERT, absl::MakeSpan(payload)));
    }
  });

  producer.join();
  consumer.join();
  EXPECT_FALSE(reader_failed.load())
      << "consumer saw a partial message: " << reader_status;
  EXPECT_EQ(read_count.load(), kMessages);
}

TEST(RingTest, OversizeMessageIsInvalidArgumentNotRingFull) {
  // "Message larger than ring capacity" is PERMANENT — retrying can never
  // succeed — while "ring full" is transient. They must have distinct
  // statuses: the server's Enqueue*S2C stashes ResourceExhausted in an outbox
  // for retry, and an oversize message would spin there forever (the
  // >capacity SAMPLE_RESP/SERVER_INFO_RESP hang).
  auto s = Ring::Create(UniqueName("oversize"), 4, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();

  std::string big(5000, 'x');  // > 4 slots x 240-byte body
  absl::Status st = ring.TryWrite(INSERT, absl::MakeSpan(big));
  EXPECT_TRUE(absl::IsInvalidArgument(st))
      << "oversize must be InvalidArgument (permanent), got " << st;

  // Transient full stays ResourceExhausted: fill all 4 slots, TryWrite a 5th.
  for (int i = 0; i < 4; i++) {
    std::string p = "m" + std::to_string(i);
    REVERB_ASSERT_OK(ring.Write(SAMPLE, absl::MakeSpan(p)));
  }
  std::string one = "x";
  absl::Status full = ring.TryWrite(INSERT, absl::MakeSpan(one));
  EXPECT_TRUE(absl::IsResourceExhausted(full))
      << "ring-full must stay ResourceExhausted (transient), got " << full;
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

TEST(RingTest, TryWriteOnFullRingReturnsResourceExhaustedWithoutBlocking) {
  auto s = Ring::Create(UniqueName("tryfull"), 4, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  // Fill all 4 slots.
  for (int i = 0; i < 4; i++) {
    std::string p = "m" + std::to_string(i);
    REVERB_ASSERT_OK(ring.Write(SAMPLE, absl::MakeSpan(p)));
  }
  // TryWrite on a full ring must return ResourceExhausted("RING_FULL")
  // immediately (no blocking), and must NOT advance head (no partial write).
  std::string p = "overflow";
  auto start = std::chrono::steady_clock::now();
  absl::Status st = ring.TryWrite(RELEASE, absl::MakeSpan(p));
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::IsResourceExhausted(st)) << st;
  EXPECT_THAT(std::string(st.message()), HasSubstr("RING_FULL"));
  // Must return near-instantly (a blocking Write would spin until a slot
  // frees, which never happens here -> test would hang). 100ms is generous.
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                .count(),
            100);
  // Drain the ring; the overflow TryWrite did not land.
  for (int i = 0; i < 4; i++) {
    MsgType type;
    std::string out;
    REVERB_ASSERT_OK(ring.Read(&type, &out));
    EXPECT_EQ(out, "m" + std::to_string(i));
  }
}

TEST(RingTest, TryWriteWithSpaceMatchesWrite) {
  auto s = Ring::Create(UniqueName("tryspace"), 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  std::string payload = "trywrite-payload";
  REVERB_ASSERT_OK(ring.TryWrite(HELLO, absl::MakeSpan(payload)));
  MsgType type;
  std::string out;
  REVERB_ASSERT_OK(ring.Read(&type, &out));
  EXPECT_EQ(type, HELLO);
  EXPECT_EQ(out, payload);
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

TEST(RingTest, OpenRejectsBadVersion) {
  auto name = UniqueName("badver");
  auto s = Ring::Create(name, 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring server = std::move(s).value();
  PokeSegment(name, Ring::TotalBytes(16, 256), [](char* base) {
    reinterpret_cast<RingHeader*>(base)->version = 999;
  });
  auto c = Ring::Open(name);
  ASSERT_FALSE(c.ok());
  EXPECT_THAT(std::string(c.status().message()), HasSubstr("version"));
}

TEST(RingTest, OpenRejectsBadGeometry) {
  // Capacity not a power of two (Ring::Create rejects the same geometry;
  // Open must not trust in-segment geometry from a stale/corrupt segment
  // either).
  auto n1 = UniqueName("badgeo_cap");
  auto s1 = Ring::Create(n1, 16, 256);
  REVERB_ASSERT_OK(s1.status());
  Ring r1 = std::move(s1).value();
  PokeSegment(n1, Ring::TotalBytes(16, 256), [](char* base) {
    reinterpret_cast<RingHeader*>(base)->capacity = 12;
  });
  EXPECT_FALSE(Ring::Open(n1).ok());

  // slot_size too small to hold even a SlotHeader.
  auto n2 = UniqueName("badgeo_slot");
  auto s2 = Ring::Create(n2, 16, 256);
  REVERB_ASSERT_OK(s2.status());
  Ring r2 = std::move(s2).value();
  PokeSegment(n2, Ring::TotalBytes(16, 256), [](char* base) {
    reinterpret_cast<RingHeader*>(base)->slot_size = 8;
  });
  EXPECT_FALSE(Ring::Open(n2).ok());
}

TEST(RingTest, ReadRejectsCorruptBodyLen) {
  auto name = UniqueName("badlen");
  auto s = Ring::Create(name, 16, 256);
  REVERB_ASSERT_OK(s.status());
  Ring ring = std::move(s).value();
  std::string payload = "abc";
  REVERB_ASSERT_OK(ring.Write(HELLO, absl::MakeSpan(payload)));
  // head/tail start at 1, so the first message lives in slot index 1.
  PokeSegment(name, Ring::TotalBytes(16, 256), [](char* base) {
    auto* slot =
        reinterpret_cast<SlotHeader*>(base + sizeof(RingHeader) + 256);
    slot->body_len = 0xFFFFFFFF;  // way beyond slot_size - sizeof(SlotHeader)
  });
  MsgType type;
  std::string out;
  absl::Status st = ring.Read(&type, &out);
  ASSERT_FALSE(st.ok());
  EXPECT_THAT(std::string(st.message()), HasSubstr("body_len"));
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
