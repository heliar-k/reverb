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
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/shm/byte_pool.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

using ::testing::HasSubstr;

// A unique SHM name per test to avoid collisions across parallel runs. Mirrors
// ring_test's UniqueName.
std::string UniqueName(const std::string& tag) {
  return "/reverb_shm_pool_test_" + tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

// Tiny slab geometry for tests: keeps the mmap segment a few KB so we don't
// allocate 1.3GB. {64, 256} with 4 blocks/slab = 4*(64+256) = 1.3KB of data.
const size_t kSmallSlabs[] = {64, 256, 1024};
const size_t kSmallBlocksPerSlab = 4;

// ── Slab selection ──

TEST(ShmBytePoolTest, SlabSelectionPicksSmallestFittingSlab) {
  auto s = ShmBytePool::Create(UniqueName("slabsel"), kSmallSlabs,
                               kSmallBlocksPerSlab);
  REVERB_ASSERT_OK(s.status());
  ShmBytePool pool = std::move(s).value();

  // 1 byte and 64 bytes both fit in the 64-byte slab.
  auto a = pool.Allocate(1);
  REVERB_ASSERT_OK(a.status());
  EXPECT_EQ(pool.block_size_at(*a), 64u);

  auto b = pool.Allocate(64);
  REVERB_ASSERT_OK(b.status());
  EXPECT_EQ(pool.block_size_at(*b), 64u);

  // 65 bytes spills into the 256-byte slab.
  auto c = pool.Allocate(65);
  REVERB_ASSERT_OK(c.status());
  EXPECT_EQ(pool.block_size_at(*c), 256u);

  auto d = pool.Allocate(256);
  REVERB_ASSERT_OK(d.status());
  EXPECT_EQ(pool.block_size_at(*d), 256u);

  // 257 bytes spills into the 1024-byte slab.
  auto e = pool.Allocate(257);
  REVERB_ASSERT_OK(e.status());
  EXPECT_EQ(pool.block_size_at(*e), 1024u);
}

TEST(ShmBytePoolTest, AllocateTooLargeFails) {
  auto s = ShmBytePool::Create(UniqueName("toolarge"), kSmallSlabs,
                               kSmallBlocksPerSlab);
  REVERB_ASSERT_OK(s.status());
  ShmBytePool pool = std::move(s).value();
  // Larger than the biggest slab (1024).
  auto a = pool.Allocate(2048);
  EXPECT_FALSE(a.ok());
  EXPECT_THAT(std::string(a.status().message()), HasSubstr("too large"));
}

// ── Allocate → write → read → Deallocate → recycle (LIFO) ──

TEST(ShmBytePoolTest, AllocateWriteReadDeallocateRecycle) {
  auto s = ShmBytePool::Create(UniqueName("recycle"), kSmallSlabs,
                               kSmallBlocksPerSlab);
  REVERB_ASSERT_OK(s.status());
  ShmBytePool pool = std::move(s).value();

  auto off = pool.Allocate(100);
  REVERB_ASSERT_OK(off.status());
  EXPECT_EQ(pool.block_size_at(*off), 256u);

  // Write bytes through At, read them back.
  std::string payload = "hello-byte-pool";
  std::memcpy(pool.At(*off), payload.data(), payload.size());
  std::string out(payload.size(), '\0');
  std::memcpy(&out[0], pool.At(*off), payload.size());
  EXPECT_EQ(out, payload);

  uint64_t first = *off;
  pool.Deallocate(*off);

  // Re-allocating the same size should recycle the just-freed block (LIFO):
  // the free list pushes to head, so the same offset comes back.
  auto off2 = pool.Allocate(100);
  REVERB_ASSERT_OK(off2.status());
  EXPECT_EQ(*off2, first);
}

// ── Refcount ──

TEST(ShmBytePoolTest, RefcountIncDecToZeroSignalsDealloc) {
  auto s = ShmBytePool::Create(UniqueName("refcount"), kSmallSlabs,
                               kSmallBlocksPerSlab);
  REVERB_ASSERT_OK(s.status());
  ShmBytePool pool = std::move(s).value();

  auto off = pool.Allocate(100);
  REVERB_ASSERT_OK(off.status());

  pool.Ref(*off);
  pool.Ref(*off);  // count = 2
  EXPECT_FALSE(pool.Unref(*off));  // count = 1, not zero
  EXPECT_TRUE(pool.Unref(*off));   // count = 0, ready for dealloc

  // Caller is responsible for deallocating once Unref returns true.
  pool.Deallocate(*off);

  // After dealloc, the block recycles.
  auto off2 = pool.Allocate(100);
  REVERB_ASSERT_OK(off2.status());
  EXPECT_EQ(*off2, *off);
}

// ── Pool-full blocking ──

TEST(ShmBytePoolTest, AllocateReturnsResourceExhaustedWhenSlabFull) {
  // Exhaustion must FAIL FAST, never block: the server's single dispatch
  // thread is the sole Allocate caller AND the sole Deallocate caller, so a
  // blocking Allocate there can never be unblocked — an unrecoverable
  // server-wide deadlock (and Stop() then hangs joining the thread).
  const size_t one_slab[] = {256};
  auto s = ShmBytePool::Create(UniqueName("fullfast"), one_slab,
                               kSmallBlocksPerSlab);
  REVERB_ASSERT_OK(s.status());
  ShmBytePool pool = std::move(s).value();

  // Exhaust the 4-block slab.
  std::vector<uint64_t> held;
  for (size_t i = 0; i < kSmallBlocksPerSlab; i++) {
    auto a = pool.Allocate(256);
    REVERB_ASSERT_OK(a.status());
    held.push_back(*a);
  }

  // The 5th allocate must return promptly with RESOURCE_EXHAUSTED. Run it on
  // a detached thread so a blocking implementation fails the assertion in 2s
  // instead of hanging the test binary.
  std::promise<absl::Status> got;
  std::future<absl::Status> fut = got.get_future();
  std::thread([&] { got.set_value(pool.Allocate(256).status()); }).detach();

  ASSERT_EQ(fut.wait_for(std::chrono::seconds(2)), std::future_status::ready)
      << "Allocate blocked on an exhausted slab — dispatch-thread deadlock";
  absl::Status st = fut.get();
  EXPECT_TRUE(absl::IsResourceExhausted(st))
      << "want RESOURCE_EXHAUSTED, got " << st;

  // Recovery: freeing a block makes the tier allocatable again (LIFO).
  pool.Deallocate(held[0]);
  auto b = pool.Allocate(256);
  REVERB_ASSERT_OK(b.status());
  EXPECT_EQ(*b, held[0]);
}

// ── Multiple blocks same slab reused correctly (no aliasing) ──

TEST(ShmBytePoolTest, MultipleBlocksSameSlabNoAliasing) {
  auto s = ShmBytePool::Create(UniqueName("noalias"), kSmallSlabs,
                               kSmallBlocksPerSlab);
  REVERB_ASSERT_OK(s.status());
  ShmBytePool pool = std::move(s).value();

  // Allocate all 4 blocks of the 64-byte slab; offsets must be distinct.
  std::vector<uint64_t> offs;
  for (size_t i = 0; i < kSmallBlocksPerSlab; i++) {
    auto a = pool.Allocate(32);
    REVERB_ASSERT_OK(a.status());
    EXPECT_EQ(pool.block_size_at(*a), 64u);
    offs.push_back(*a);
  }
  // Distinct, non-overlapping offsets.
  for (size_t i = 0; i < offs.size(); i++)
    for (size_t j = i + 1; j < offs.size(); j++)
      EXPECT_NE(offs[i], offs[j]);

  // Write a distinct byte to each, confirm no aliasing.
  for (size_t i = 0; i < offs.size(); i++) {
    *reinterpret_cast<char*>(pool.At(offs[i])) = static_cast<char>('A' + i);
  }
  for (size_t i = 0; i < offs.size(); i++) {
    EXPECT_EQ(*reinterpret_cast<char*>(pool.At(offs[i])),
              static_cast<char>('A' + i));
  }

  // Exhausting the slab fails fast (see AllocateReturnsResourceExhausted-
  // WhenSlabFull); freeing one block recycles it.
  auto full = pool.Allocate(32);
  EXPECT_TRUE(absl::IsResourceExhausted(full.status())) << full.status();
  pool.Deallocate(offs[0]);
  auto recycled = pool.Allocate(32);
  REVERB_ASSERT_OK(recycled.status());
  EXPECT_EQ(*recycled, offs[0]);
}

// ── ReleaseAll ──

TEST(ShmBytePoolTest, ReleaseAllDeallocatesRefcountedOffsets) {
  auto s = ShmBytePool::Create(UniqueName("releaseall"), kSmallSlabs,
                               kSmallBlocksPerSlab);
  REVERB_ASSERT_OK(s.status());
  ShmBytePool pool = std::move(s).value();

  // Allocate 3 blocks, Ref each once (refcount = 1).
  std::vector<uint64_t> offs;
  for (int i = 0; i < 3; i++) {
    auto a = pool.Allocate(100);
    REVERB_ASSERT_OK(a.status());
    pool.Ref(*a);
    offs.push_back(*a);
  }

  // Snapshot the offsets; ReleaseAll should drive each to refcount 0 and
  // deallocate them, so re-allocating the same sizes recycles the same offsets
  // (in LIFO order).
  pool.ReleaseAll(offs);

  std::vector<uint64_t> recycled;
  for (int i = 0; i < 3; i++) {
    auto a = pool.Allocate(100);
    REVERB_ASSERT_OK(a.status());
    recycled.push_back(*a);
  }
  // The recycled set equals the released set (order may differ due to LIFO).
  std::set<uint64_t> released_set(offs.begin(), offs.end());
  std::set<uint64_t> recycled_set(recycled.begin(), recycled.end());
  EXPECT_EQ(released_set, recycled_set);
}

// ── Cross-thread (same process) mmap sharing ──

TEST(ShmBytePoolTest, ServerWritesClientReadsAcrossMappings) {
  // Use two threads acting as server/client. Both mmap the same SHM segment
  // (MAP_SHARED), so bytes the server writes at a granted offset are visible
  // to the client's mapping. This proves the pool crosses the process
  // boundary (single process here is enough to prove mmap sharing).
  const std::string name = UniqueName("xthread");

  constexpr char kMarker[] = "cross-mapping-payload-1234567890";
  const size_t kLen = sizeof(kMarker);  // incl. NUL → fits 64-byte slab

  // Two separate handoffs: offset (server→client), then done (client→server).
  std::promise<uint64_t> off_promise;
  std::future<uint64_t> off_future = off_promise.get_future();
  std::promise<void> done_promise;
  std::shared_future<void> done_future = done_promise.get_future().share();

  std::thread server([&] {
    auto s = ShmBytePool::Create(name, kSmallSlabs, kSmallBlocksPerSlab);
    ASSERT_TRUE(s.ok()) << s.status();
    ShmBytePool pool = std::move(s).value();
    auto off = pool.Allocate(kLen);
    ASSERT_TRUE(off.ok()) << off.status();
    std::memcpy(pool.At(*off), kMarker, kLen);
    off_promise.set_value(*off);
    // Keep the pool alive until the client is done reading.
    done_future.wait();
  });

  uint64_t offset = off_future.get();

  // Client opens the same segment (RW per C4) and reads at the granted offset.
  auto co = ShmBytePool::Open(name);
  REVERB_ASSERT_OK(co.status());
  ShmBytePool client_pool = std::move(co).value();
  char buf[kLen];
  std::memcpy(buf, client_pool.At(offset), kLen);
  EXPECT_EQ(std::string(buf), std::string(kMarker));
  // Signal the server it can tear down.
  done_promise.set_value();

  server.join();
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
