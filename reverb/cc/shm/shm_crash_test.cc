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

// Ticket ⑥: crash recovery + cleanup reliability. A client whose udsocket fd
// closes (crash or graceful close) must be reclaimed by the server — its
// outstanding pool offsets centrally released (C3), its two rings shm_unlinked
// (R6), its ClientState erased — WITHOUT affecting other clients and while the
// server keeps running. A new client must reconnect cleanly afterward.
//
// ponytail: we simulate a crash by closing the client's control_fd
// (ShmConnection::control_fd), which is exactly the liveness signal the server
// poll()s (spec §8.8). This avoids fork/SIGKILL and the PID-collision issue
// (two in-process clients share getpid() -> identical ring names -> second
// Ring::Create would clobber the first). For the "other client unaffected"
// case we fork a real child process (distinct PID -> distinct ring names) and
// _exit() it abruptly. Same byte/shape assertion rationale as shm_sample_test
// (no CPython interpreter in this binary).

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <string>
#include <sys/mman.h>  // shm_open
#include <sys/wait.h>  // waitpid
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_server.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "reverb/cc/tensor_compression.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

using ::testing::SizeIs;

// ── Table-seeding helpers (mirrors shm_sample_test.cc / sampler_test.cc) ──

SequenceRange MakeSequenceRange(uint64_t episode_id, int32_t start,
                                int32_t end) {
  REVERB_CHECK_LE(start, end);
  SequenceRange range;
  range.set_episode_id(episode_id);
  range.set_start(start);
  range.set_end(end);
  return range;
}

PrioritizedItem MakePrioritizedItem(uint64_t key, double priority,
                                    const std::vector<ChunkData>& chunks) {
  REVERB_CHECK(!chunks.empty());
  PrioritizedItem item;
  item.set_key(key);
  item.set_priority(priority);
  for (int i = 0; i < chunks.front().data().tensors_size(); ++i) {
    auto* col = item.mutable_flat_trajectory()->add_columns();
    for (const auto& chunk : chunks) {
      auto* slice = col->add_chunk_slices();
      slice->set_chunk_key(chunk.chunk_key());
      slice->set_offset(0);
      slice->set_length(chunk.data().tensors(i).shape().dim(0));
      slice->set_index(i);
    }
  }
  return item;
}

TensorBuffer MakeTensor(int length) {
  std::vector<uint64_t> values(length * 2);
  for (int i = 0; i < length * 2; i++) values[i] = static_cast<uint64_t>(i);
  std::string bytes(values.size() * sizeof(uint64_t), '\0');
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return TensorBuffer(TensorSpec{DataType::Uint64, {length, 2}},
                      std::move(bytes));
}

ChunkData MakeChunkData(uint64_t key, SequenceRange range) {
  ChunkData chunk;
  chunk.set_chunk_key(key);
  auto t = MakeTensor(range.end() - range.start() + 1);
  CHECK_OK(
      CompressTensorAsProto(t, chunk.mutable_data()->add_tensors()));
  *chunk.mutable_sequence_range() = std::move(range);
  return chunk;
}

TableItem MakeItem(uint64_t key, double priority,
                   const std::vector<SequenceRange>& sequences, int32_t offset,
                   int32_t length) {
  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
  std::vector<ChunkData> data(sequences.size());
  for (int i = 0; i < sequences.size(); i++) {
    data[i] = MakeChunkData(key * 100 + i, sequences[i]);
    chunks.push_back(std::make_shared<ChunkStore::Chunk>(data[i]));
  }
  Table::Item item(MakePrioritizedItem(key, priority, data), std::move(chunks));

  int32_t remaining = length;
  for (int slice_index = 0; slice_index < sequences.size(); slice_index++) {
    for (int col_index = 0;
         col_index < item.flat_trajectory().columns_size(); col_index++) {
      auto* col =
          item.unsafe_mutable_flat_trajectory()->mutable_columns(col_index);
      auto* slice = col->mutable_chunk_slices(slice_index);
      slice->set_offset(offset);
      slice->set_length(
          std::min<int32_t>(slice->length() - slice->offset(), remaining));
      slice->set_index(col_index);
    }
    remaining -=
        item.flat_trajectory().columns(0).chunk_slices(slice_index).length();
    offset = 0;
  }
  return item;
}

void InsertItem(Table* table, uint64_t key, double priority,
                std::vector<int> sequence_lengths, int32_t offset = 0,
                int32_t length = 0, bool squeeze = false) {
  REVERB_CHECK(!squeeze || length == 1);
  if (length == 0) {
    length =
        std::accumulate(sequence_lengths.begin(), sequence_lengths.end(), 0) -
        offset;
  }
  std::vector<SequenceRange> ranges(sequence_lengths.size());
  int step_index = 0;
  for (int i = 0; i < sequence_lengths.size(); i++) {
    ranges[i] = MakeSequenceRange(100 * key, step_index,
                                  step_index + sequence_lengths[i] - 1);
    step_index += sequence_lengths[i];
  }
  auto item = MakeItem(key, priority, ranges, offset, length);
  item.unsafe_mutable_flat_trajectory()->mutable_columns(0)->set_squeeze(
      squeeze);
  REVERB_EXPECT_OK(table->InsertOrAssign(std::move(item)));
}

std::shared_ptr<Table> MakeTable(int max_size = 100) {
  return std::make_shared<Table>(
      /*name=*/"queue",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, max_size));
}

std::string UniqueTag(const std::string& tag) {
  return tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

// Try to shm_open a ring name; returns true if the segment still EXISTS
// (i.e. was NOT unlinked). Used to assert HandleDisconnect unlinked the rings.
bool ShmSegmentExists(const std::string& name) {
  int fd = shm_open(name.c_str(), O_RDWR, 0600);
  if (fd >= 0) {
    close(fd);
    return true;
  }
  return false;
}

// Poll until `cond()` is true or `deadline` passes. The server's disconnect
// detection happens on its dispatch thread, so the test must wait a few loop
// iterations for HandleDisconnect to fire after the client fd closes.
bool WaitFor(std::function<bool()> cond, absl::Duration timeout) {
  absl::Time deadline = absl::Now() + timeout;
  while (absl::Now() < deadline) {
    if (cond()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return cond();
}

// ── Crash recovery (single client) ──

// A client samples (so the server holds outstanding_offsets_ for it), then its
// control fd closes (simulating a crash). The server must: reclaim the
// offsets (C3), unlink the two rings (R6), erase the client, and keep running
// so a fresh client reconnects + samples cleanly.
TEST(ShmCrashTest, ClientFdCloseReclaimsOffsetsAndUnlinksRings) {
  auto table = MakeTable();
  // Two items: one sampled by the crashed client, one for the reconnecting
  // client. (MakeTable uses max_times_sampled=1, so a single item could not
  // serve two samples.)
  InsertItem(table.get(), /*key=*/1, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);
  InsertItem(table.get(), /*key=*/2, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);

  std::string sock = "/tmp/reverb_shm_crash_" + UniqueTag("c1") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK((*client)->NewSampler("queue", {1}, &sampler));

  // One sample: the server allocates a pool offset + Refs it (outstanding).
  std::vector<TensorBuffer> data;
  REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_THAT(data, SizeIs(1));
  // NOTE: we deliberately do NOT close the sampler / do not send RELEASE —
  // the offset stays outstanding, as it would if the client crashed mid-use.

  // The client's liveness fd. Closing it is the crash signal (spec §8.8).
  ShmConnection* conn = (*client)->connection();
  ASSERT_GE(conn->control_fd, 0);
  // Recompute this client's ring names to assert unlink later. The client sent
  // getpid() as its PID; the server's PID is also getpid() (same process).
  ShmSegmentNames names = MakeShmNames(sock, getpid());
  // Sanity: all four rings exist while the client is connected (decision D).
  ASSERT_TRUE(ShmSegmentExists(names.insert_c2s));
  ASSERT_TRUE(ShmSegmentExists(names.insert_s2c));
  ASSERT_TRUE(ShmSegmentExists(names.sample_c2s));
  ASSERT_TRUE(ShmSegmentExists(names.sample_s2c));

  // Simulate crash: close the control fd. The server's dispatch loop poll()s
  // it next pass, sees EOF, and calls HandleDisconnect.
  close(conn->control_fd);
  conn->control_fd = -1;  // prevent ~ShmConnection double-close

  // Wait for the server to unlink all four rings (HandleDisconnect ran).
  ASSERT_TRUE(WaitFor([&] { return !ShmSegmentExists(names.insert_c2s); },
                      absl::Seconds(5)))
      << "server did not unlink insert c2s ring after client fd close";
  EXPECT_FALSE(ShmSegmentExists(names.insert_s2c))
      << "server did not unlink insert s2c ring after client fd close";
  EXPECT_FALSE(ShmSegmentExists(names.sample_c2s))
      << "server did not unlink sample c2s ring after client fd close";
  EXPECT_FALSE(ShmSegmentExists(names.sample_s2c))
      << "server did not unlink sample s2c ring after client fd close";

  // Drop the client objects. The rings are already unlinked by the server;
  // ~Ring's owner-unlink is a harmless ENOENT.
  sampler->Close();
  client->reset();

  // The server must keep running: a fresh client connects + samples.
  auto client2 = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client2.status());
  std::unique_ptr<ShmSampler> sampler2;
  REVERB_ASSERT_OK((*client2)->NewSampler("queue", {1}, &sampler2));
  std::vector<TensorBuffer> data2;
  REVERB_EXPECT_OK(sampler2->GetNextTrajectory(&data2));
  ASSERT_THAT(data2, SizeIs(1));
  EXPECT_EQ(data2[0].shape()[0], 5);

  sampler2->Close();
  (*server)->Stop();
}

// ── Offsets reclaimed: no pool exhaustion after a crash ──

// If HandleDisconnect did NOT ReleaseAll the crashed client's outstanding
// offsets, the pool tier would leak one block per crashed-and-reconnected
// client. We crash+reconnect more times than a single pool tier holds blocks
// (kDefaultBlocksPerSlab = 256; each [5,2] uint64 sample = 80 bytes -> 256-byte
// tier). If reclamation is broken, the 257th allocation blocks forever on the
// pool condvar and the test TIMES OUT. If reclamation works, all succeed.
TEST(ShmCrashTest, RepeatedCrashDoesNotExhaustPool) {
  constexpr int kRounds = 300;  // > 256-block tier
  auto table = MakeTable(kRounds);
  for (int i = 1; i <= kRounds; i++) {
    InsertItem(table.get(), /*key=*/i, /*priority=*/1.0,
               /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);
  }

  std::string sock = "/tmp/reverb_shm_crash_" + UniqueTag("pool") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  // Each round: connect, sample (offset outstanding), crash without RELEASE,
  // wait for the server to reclaim. If reclamation leaks, the pool tier
  // exhausts after 256 rounds and round 257's server-side Allocate blocks.
  for (int i = 0; i < kRounds; i++) {
    auto client = ShmClient::Connect(sock);
    REVERB_ASSERT_OK(client.status());
    std::unique_ptr<ShmSampler> sampler;
    REVERB_ASSERT_OK((*client)->NewSampler("queue", {1}, &sampler));
    std::vector<TensorBuffer> data;
    REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data)) << "round " << i;

    // Crash: close the control fd, wait for the server to reclaim.
    ShmConnection* conn = (*client)->connection();
    close(conn->control_fd);
    conn->control_fd = -1;
    // Wait for the server to detect the disconnect. We can't observe
    // outstanding_offsets_ directly (private), but the next round's Connect
    // would fail/block if the server were wedged. Poll the ring unlink as the
    // reclamation-completed signal: ring names are getpid()/getpid() each
    // round (same process), so the server must unlink before the next round's
    // Ring::Create(names.insert_c2s) can succeed with O_EXCL.
    ShmSegmentNames names = MakeShmNames(sock, getpid());
    ASSERT_TRUE(WaitFor([&] { return !ShmSegmentExists(names.insert_c2s); },
                        absl::Seconds(5)))
        << "round " << i << ": server did not reclaim after crash";
    sampler->Close();
    client->reset();
  }

  (*server)->Stop();
}

// ── Other client unaffected (forked child = distinct PID) ──

// ponytail: two clients in ONE process collide on ring names (both send
// getpid()), so the "other client unaffected" case forks a real child process
// (distinct PID -> distinct ring names). The child connects, samples, then
// _exit()s abruptly (crash). The parent asserts: (a) the parent-process
// client can still sample while/after the child crashed, (b) a fresh child
// reconnects.
TEST(ShmCrashTest, OtherClientUnaffectedByChildCrash) {
  auto table = MakeTable();
  // Three items: parent samples twice (max_samples=2, pre-fetches both), the
  // child samples once. max_times_sampled=1 means each item serves one sample.
  InsertItem(table.get(), /*key=*/1, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);
  InsertItem(table.get(), /*key=*/2, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);
  InsertItem(table.get(), /*key=*/3, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);

  std::string sock = "/tmp/reverb_shm_crash_" + UniqueTag("multi") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  // Parent-side client: must keep working throughout.
  auto parent_client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(parent_client.status());
  std::unique_ptr<ShmSampler> parent_sampler;
  REVERB_ASSERT_OK((*parent_client)->NewSampler("queue", {2}, &parent_sampler));
  std::vector<TensorBuffer> data0;
  REVERB_EXPECT_OK(parent_sampler->GetNextTrajectory(&data0));
  ASSERT_THAT(data0, SizeIs(1));

  // Fork a child that connects, samples, then crashes (_exit).
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    // Child: connect, sample, then die abruptly (no RELEASE, no Close).
    auto c = ShmClient::Connect(sock);
    if (!c.ok()) _exit(1);
    std::unique_ptr<ShmSampler> s;
    if (!(*c)->NewSampler("queue", {1}, &s).ok()) _exit(2);
    std::vector<TensorBuffer> d;
    if (!s->GetNextTrajectory(&d).ok()) _exit(3);
    _exit(0);  // crash: control_fd closes, rings (child-PID-named) unlinked by server
  }
  // Parent: wait for the child to exit.
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
      << "child did not sample successfully";

  // Give the server a moment to detect the child's fd EOF + reclaim.
  ShmSegmentNames child_names = MakeShmNames(sock, pid);
  ASSERT_TRUE(WaitFor([&] { return !ShmSegmentExists(child_names.insert_c2s); },
                      absl::Seconds(5)))
      << "server did not unlink crashed child's insert c2s ring";

  // The parent client must STILL be able to sample (unaffected by the crash).
  std::vector<TensorBuffer> data1;
  REVERB_EXPECT_OK(parent_sampler->GetNextTrajectory(&data1))
      << "parent client broke after child crash";
  ASSERT_THAT(data1, SizeIs(1));

  parent_sampler->Close();
  (*server)->Stop();
}

// ── Client-side EOF: server gone while a sample is in flight (ticket ⑥) ──

// ticket ⑥ spec §8.8: if the server dies/closes while the client is waiting
// for an S→C response, the client must detect it via the liveness control_fd
// and fail fast (UnavailableError -> reverb.errors.ConnectionError on the
// Python side) instead of spinning forever on the ring. A regression that
// reverts the control_fd probe would hang ReadBlocking indefinitely.
//
// Setup: one seeded item, sampler max_samples=2. The worker pre-fetches the
// first sample, then sends a second SAMPLE and blocks in ReadBlocking on the
// S→C ring (no second item exists, so no response is forthcoming). We then
// close the SERVER's accepted fd (CloseClientFdForTest) — the peer of the
// client's control_fd — which is exactly "the server side of the connection
// dropped" (spec §8.8 client path). ReadBlocking's IsPeerClosed(control_fd)
// must see the EOF and return UnavailableError, closing the sampler's queue so
// GetNextTrajectory unblocks with a non-OK status. The wait is bounded by a
// std::future so a regression hang fails the TEST (timeout) not the harness.
//
// We do NOT use server->Stop() to drop the fd: Stop() joins the dispatch
// thread, which may be blocked in Table::Sample on the second request. Instead
// we close the fd directly, then Close()+Stop() for cleanup (Close unblocks any
// in-flight Table::Sample via stop_worker_ so Stop()'s join completes).
TEST(ShmCrashTest, ClientFailsFastWhenServerStops) {
  auto table = MakeTable();
  // One item: the worker fetches it, then blocks in ReadBlocking on the
  // second sample (no item to serve -> no response -> pure EOF path).
  InsertItem(table.get(), /*key=*/1, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);

  std::string sock = "/tmp/reverb_shm_crash_" + UniqueTag("eof") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());
  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK((*client)->NewSampler("queue", {2}, &sampler));

  // First sample succeeds (pre-fetched by the worker).
  std::vector<TensorBuffer> data;
  REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_THAT(data, SizeIs(1));

  // Give the worker a moment to send the second SAMPLE and enter ReadBlocking.
  // (A brief sleep is fine: the worker sends the request then spins in
  // ReadBlocking; we just need it to be waiting before we drop the fd.)
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Drop the server side of the connection ("server closed/crashed"). The
  // client's control_fd peer is now gone.
  (*server)->CloseClientFdForTest();

  // The second GetNextTrajectory must return a non-OK status (the worker's
  // ReadBlocking saw control_fd EOF) within a BOUNDED time. Run it on a
  // separate thread + future so a hang fails the test instead of hanging it.
  auto fut = std::async(std::launch::async, [&] {
    std::vector<TensorBuffer> d;
    return sampler->GetNextTrajectory(&d).code();
  });
  ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready)
      << "client hung waiting for a response after the server closed "
      << "(control_fd EOF probe missing in ReadBlocking?)";
  EXPECT_EQ(fut.get(), absl::StatusCode::kUnavailable)
      << "expected UnavailableError (server closed), got a different status";

  sampler->Close();
  // Cleanup: Close unblocks any in-flight server-side Table::Sample so the
  // dispatch thread's join in Stop() completes.
  table->Close();
  (*server)->Stop();
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
