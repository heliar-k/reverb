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
// Review #1 (shm-clientstate-lifetime): the last two tests keep insert/sample
// callbacks pending on a rate-limiter-blocked table while Stop()/disconnect
// tears down ClientState — a pre-fix heap-use-after-free, caught by ASan/TSan.
//
// ponytail: we simulate a crash by closing the client's control_fd
// (ShmConnection::control_fd), which is exactly the liveness signal the server
// poll()s (spec §8.8). This avoids fork/SIGKILL and the PID-collision issue
// (two in-process clients share getpid() -> identical ring names -> second
// Ring::Create would clobber the first). For the "other client unaffected"
// case we fork a real child process (distinct PID -> distinct ring names) and
// _exit() it abruptly. Same byte/shape assertion rationale as shm_sample_test
// (no CPython interpreter in this binary).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dirent.h>  // opendir/readdir
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <sys/mman.h>  // shm_open
#include <sys/stat.h>  // fstat
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
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/shm/bootstrap.h"
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_server.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"
#include "reverb/cc/tensor_compression.h"
#include "reverb/cc/trajectory_writer.h"
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

// The four ring names the server assigned this connection via Welcome. Ticket
// #7: names carry a per-server epoch, so tests can no longer recompute them
// from socket path + PID — read them off the connection instead.
ShmSegmentNames NamesFromConn(const ShmConnection& conn) {
  ShmSegmentNames names;
  names.pool = conn.pool_shm_name;
  names.insert_c2s = conn.insert_c2s.shm_name();
  names.insert_s2c = conn.insert_s2c.shm_name();
  names.sample_c2s = conn.sample_c2s.shm_name();
  names.sample_s2c = conn.sample_s2c.shm_name();
  return names;
}

// Count segments in /dev/shm whose name ends with `suffix`. For asserting a
// forked CHILD's rings were reclaimed when the parent can't read the child's
// connection (ticket #7: epoch-keyed names, matched here by `_<pid>` suffix).
int CountShmSegmentsWithSuffix(const std::string& suffix) {
  DIR* dir = opendir("/dev/shm");
  if (dir == nullptr) return 0;
  int n = 0;
  while (dirent* e = readdir(dir)) {
    std::string name = e->d_name;
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      n++;
    }
  }
  closedir(dir);
  return n;
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

// ── ClientState lifetime (ticket: shm-clientstate-lifetime, review #1) ──
//
// A table whose rate limiter blocks BOTH inserts and samples forever keeps
// every async insert/sample callback pending until the table worker shuts
// down. Stop()/HandleDisconnect then deterministically race the shutdown-time
// callbacks with ClientState teardown — pre-fix that was a heap-use-after-free
// (clients_ cleared/erased while table workers still fired callbacks that
// dereference raw ClientState*; caught by ASan/TSan). Post-fix the callbacks
// hold shared_ptr<ClientState> and ShmServer::Stop stops tables before
// clearing clients.
std::shared_ptr<Table> MakeBlockingTable() {
  // RateLimiter(1, 1, min_diff=1, max_diff=0): the first insert is free
  // (inserts+1 <= min_size_to_sample), everything after is blocked:
  // CanInsert diff=(1+n)*1-0 > 0 for n>=1, and CanSample diff=1*1-0-1=0 <
  // min_diff=1. The caller seeds ONE item synchronously to consume the free
  // insert, after which every async insert/sample stays pending until worker
  // shutdown.
  return std::make_shared<Table>(
      /*name=*/"queue",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/100,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, /*min_diff=*/1,
                                                     /*max_diff=*/0));
}

using Step = std::vector<std::optional<TensorBuffer>>;
using StepRef = std::vector<std::optional<std::weak_ptr<CellRef>>>;

TensorBuffer MakeZeroInt32() {
  int32_t zero = 0;
  std::string bytes(sizeof(int32_t), '\0');
  std::memcpy(bytes.data(), &zero, sizeof(int32_t));
  return TensorBuffer(TensorSpec{DataType::Int32, {1}}, std::move(bytes));
}

// Fire-and-forget insert on a background thread: Append + CreateItem + Flush.
// Flush blocks awaiting INSERT_ACK, which with a blocking table never comes —
// the call returns (with an error) only when the connection drops at
// Stop/disconnect. The server-side insert callback stays pending until the
// table worker shuts down.
void InsertOneItemAsync(ShmClient* client) {
  std::unique_ptr<TrajectoryWriter> writer;
  if (!client
           ->NewTrajectoryWriter(
               TrajectoryWriter::Options{
                   .chunker_options =
                       std::make_shared<ConstantChunkerOptions>(1, 1)},
               &writer)
           .ok()) {
    return;
  }
  StepRef refs;
  if (!writer->Append(Step({MakeZeroInt32()}), &refs).ok()) return;
  std::vector<std::weak_ptr<CellRef>> col{refs[0].value()};
  if (!writer->CreateItem("queue", 1.0, {TrajectoryColumn(col, /*squeeze=*/false)})
           .ok()) {
    return;
  }
  (void)writer->Flush();  // errors on connection close; ignored
}

// Fire-and-forget sample on a background thread. The empty blocking table
// never serves the request (infinite rate_limiter_timeout -> no server-side
// deadline), so the completion callback stays pending until table shutdown;
// the client call errors out when the connection drops.
void SampleOneItemAsync(ShmClient* client) {
  std::unique_ptr<ShmSampler> sampler;
  if (!client->NewSampler("queue", {1}, &sampler).ok()) return;
  std::vector<TensorBuffer> data;
  (void)sampler->GetNextTrajectory(&data);  // blocks; errors on conn close
  sampler->Close();
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
  // Read this client's ring names off the connection to assert unlink later.
  ShmSegmentNames names = NamesFromConn(*conn);
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
    // reclamation-completed signal: ring names are identical each round
    // (same server epoch + same process PID), so the server must unlink
    // before the next round's Ring::Create(names.insert_c2s) can succeed
    // with O_EXCL.
    ShmSegmentNames names = NamesFromConn(*conn);
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

  // Give the server a moment to detect the child's fd EOF + reclaim. The
  // parent can't read the child's connection, so match ring names by
  // client-PID suffix (ticket #7: epoch-keyed names).
  const std::string child_suffix = absl::StrCat("_", pid);
  ASSERT_TRUE(
      WaitFor([&] { return CountShmSegmentsWithSuffix(child_suffix) == 0; },
              absl::Seconds(5)))
      << "server did not unlink crashed child's rings";

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

// ── ClientState lifetime: Server.stop() with in-flight callbacks (review #1) ──

// An insert and a sample are in flight (callbacks pending on the blocking
// table) when the server stops. Pre-fix, Stop() cleared clients_ while the
// table workers were still alive; the pending callbacks then fired at ~Table
// (NotifyPendingInserts / FinalizeSampleRequest on the callback executor) and
// dereferenced the freed ClientState — a heap-use-after-free under ASan/TSan.
// Post-fix, Stop() stops the tables (Close + join worker + drain callback
// executor) BEFORE clearing clients_, and the callbacks hold
// shared_ptr<ClientState>. Without sanitizers this exercises the same path
// and asserts the client threads fail fast instead of hanging.
TEST(ShmCrashTest, StopWithInFlightCallbacksDoesNotUaf) {
  auto table = MakeBlockingTable();
  // Seed one item to consume the rate limiter's free first insert, so the
  // client's insert below stays pending (and the seeded item keeps the
  // sample request servable-but-rate-limited rather than unservable).
  InsertItem(table.get(), /*key=*/1, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);
  std::string sock = "/tmp/reverb_shm_crash_" + UniqueTag("stop") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());

  std::thread writer_thread([&] { InsertOneItemAsync(client->get()); });
  std::thread sampler_thread([&] { SampleOneItemAsync(client->get()); });

  // Let the dispatch thread enqueue both requests server-side. A dispatch
  // pass costs ~50us; 300ms is thousands of passes, after which both
  // callbacks are pending on the blocking table with overwhelming margin.
  absl::SleepFor(absl::Milliseconds(300));

  // Stop with both callbacks in flight: must stop tables before clearing
  // clients, and must unblock the client threads (fd close -> EOF).
  (*server)->Stop();
  writer_thread.join();
  sampler_thread.join();

  EXPECT_EQ(table->size(), 1)
      << "only the seeded item may land; the client insert stays pending";
  // ~ShmServer re-enters Stop() (idempotent) and destroys the table.
}

// ── The actual UAF repro: Server.stop() mid-callback-stream (statistical) ──
//
// Why the Stop path and not disconnect: the dispatch dead-check only runs at
// the top of a pass, so HandleDisconnect lands one full drain pass after the
// fd close (measured: 3-11ms under ASan) — by then any callback stream has
// ended, structurally. Stop() has no such delay: the dispatch join only
// waits for the CURRENT pass, and clients_.clear() runs immediately after,
// at a moment the test thread picks. The race window (review #1) is a
// callback that already lock()ed its keepalive and is mid-body (touching
// state->pending_samples_mu) when clear() frees the ClientState.
//
// The window is sub-microsecond, so the test engineers a long, hot callback
// stream and lands clear() in its middle:
//  - 8 tables share the one client's ClientState; each has its own worker +
//    callback executor -> 8 parallel callback streams into one ClientState.
//  - A SlowSampleExtension (OnSample busy-spins 300us) throttles each worker
//    to 300us/sample via WaitForBackgroundWork (extension queue cap 10), so
//    the stream outlives the dispatch drain pass (throttle > per-request
//    enqueue cost) and the pool never exhausts pre-clear (completions < 256).
//  - Stop() is called mid-stream; the dispatch join returns as the drain
//    pass ends, with most of each worker's backlog still pending — clear()
//    then races 8 hot streams. 300 rounds make the pre-fix failure
//    near-certain under ASan/TSan (duty per event is only a few %%,
//    inflated by ASan instrumentation of the body).
// Post-fix, Stop() stops the tables (Close + join worker + drain executor)
// BEFORE clearing clients, and callbacks hold shared_ptr<ClientState>.
class SlowSampleExtension : public TableExtension {
 public:
  bool CanRunAsync() const override { return true; }
  std::string DebugString() const override { return "SlowSampleExtension"; }

 protected:
  void OnInsert(absl::Mutex*, const ExtensionItem&) override {}
  void OnDelete(absl::Mutex*, const ExtensionItem&) override {}
  void OnUpdate(absl::Mutex*, const ExtensionItem&) override {}
  void OnSample(absl::Mutex*, const ExtensionItem&) override {
    absl::Time end = absl::Now() + absl::Microseconds(300);
    while (absl::Now() < end) {
    }
  }
  void OnReset(absl::Mutex*) override {}
  absl::Status RegisterTable(absl::Mutex*, Table*) override {
    return absl::OkStatus();
  }
  void UnregisterTable(absl::Mutex*, Table*) override {}
};

//
// The shutdown-path tests above CANNOT hit the bug: when ClientState is
// destroyed, the keepalive callbacks (its members) die too, so the table's
// weak_ptr.lock() fails and no callback body ever runs on the freed state.
// The real window (review #1) is a callback that ALREADY lock()ed its
// keepalive and is mid-body (touching state->pending_samples_mu) when
// HandleDisconnect frees the ClientState. That window is sub-microsecond
// (the self-erase scan is O(1) in FIFO completion order), so ONE sample
// stream gives only a few %% hit chance per erase. This test multiplies the
// odds: EIGHT tables share the one client's ClientState — each has its own
// worker + callback executor, so a raw SAMPLE blast across all 8 keeps 8
// callback streams firing in parallel, all touching the same ClientState.
// The disconnect lands while the workers grind their backlogs; 30 rounds
// make the pre-fix failure near-certain under ASan/TSan. Post-fix the
// callback's shared_ptr keeps the ClientState alive through the body.
//
// Sizing: the blast must stay UNDER the pool's 256-block tier — every
// completed sample holds one pool block outstanding until the client
// RELEASEs (this test never does), so a blast >256 exhausts the tier and
// every subsequent DrainPendingSamples fails with a per-item log line,
// inflating the dispatch pass to tens of ms. The erase then lands long
// after the callback stream ended and the race window is never sampled.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
constexpr int kStormRounds = 300;
#else
constexpr int kStormRounds = 5;  // plain builds: mechanics only, keep it fast
#endif

TEST(ShmCrashTest, StopDuringSampleCallbackStormDoesNotUaf) {
  constexpr int kNumTables = 8;
  constexpr int kBlastPerTable = 10;

  // One request body per table (raw blast: no ShmSampler, whose worker
  // thread would make sample_c2s multi-producer).
  std::vector<std::string> bodies(kNumTables);
  for (int t = 0; t < kNumTables; t++) {
    ShmSampleRequest req;
    req.set_table(absl::StrCat("t", t));
    req.set_num_samples(1);
    req.set_timeout_ms(-1);  // infinite: never expires server-side
    bodies[t] = req.SerializeAsString();
  }

  for (int round = 0; round < kStormRounds; round++) {
    // Fresh tables per round: Stop() stops them permanently (post-fix), and
    // each round's items are sampled out anyway.
    std::vector<std::shared_ptr<Table>> tables;
    for (int t = 0; t < kNumTables; t++) {
      auto table = std::make_shared<Table>(
          /*name=*/absl::StrCat("t", t),
          /*sampler=*/std::make_shared<FifoSelector>(),
          /*remover=*/std::make_shared<FifoSelector>(),
          /*max_size=*/100,
          /*max_times_sampled=*/1,  // every sample also DeleteItems
          /*rate_limiter=*/std::make_shared<RateLimiter>(
              1, 1, /*min_diff=*/-1e9, /*max_diff=*/1e9),  // never blocks
          /*extensions=*/
          std::vector<std::shared_ptr<TableExtension>>{
              std::make_shared<SlowSampleExtension>()});
      // One item per blast request: each is sampled (and deleted) once.
      for (int k = 1; k <= kBlastPerTable; k++) {
        InsertItem(table.get(), /*key=*/k, /*priority=*/1.0,
                   /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);
      }
      tables.push_back(std::move(table));
    }

    std::string sock =
        "/tmp/reverb_shm_crash_" + UniqueTag(absl::StrCat("storm", round)) +
        ".sock";
    auto server = ShmServer::Create(std::move(tables), sock);
    REVERB_ASSERT_OK(server.status());
    REVERB_ASSERT_OK((*server)->Start());
    auto client = ShmClient::Connect(sock);
    REVERB_ASSERT_OK(client.status());
    ShmConnection* conn = (*client)->connection();

    // Blast all 8 tables interleaved. The dispatch drains the ring over the
    // next few hundred us; the throttled workers (100us/sample) grind their
    // 10-deep backlogs for ~3ms, their executors firing completion
    // callbacks — all into ONE ClientState.
    for (int i = 0; i < kBlastPerTable; i++) {
      for (int t = 0; t < kNumTables; t++) {
        REVERB_ASSERT_OK(
            conn->sample_c2s.Write(SAMPLE, absl::MakeSpan(bodies[t])));
      }
    }
    // Land clients_.clear() mid-stream: the dispatch join in Stop() returns
    // as the drain pass ends (~250us from now), with most of each worker's
    // backlog — and all 8 callback streams — still hot.
    absl::SleepFor(absl::Microseconds(250));
    (*server)->Stop();
    client->reset();
  }
}

// ── ClientState lifetime: HandleDisconnect with in-flight callbacks ──

// Per-client teardown (HandleDisconnect) cannot stop the table — other
// clients share it. Each round leaves one insert + one sample callback
// pending on the blocking table whose captured ClientState is erased at
// disconnect; Stop() then fires every accumulated callback at table
// shutdown. Pre-fix each of those touched a ClientState freed rounds earlier
// (ASan/TSan); post-fix the callbacks' shared_ptr keeps the state alive
// until they return.
TEST(ShmCrashTest, DisconnectWithInFlightCallbacksDoesNotUaf) {
  constexpr int kRounds = 10;
  auto table = MakeBlockingTable();
  InsertItem(table.get(), /*key=*/1, /*priority=*/1.0,
             /*sequence_lengths=*/{5}, /*offset=*/0, /*length=*/5);
  std::string sock = "/tmp/reverb_shm_crash_" + UniqueTag("disc") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());
  for (int round = 0; round < kRounds; round++) {
    auto client = ShmClient::Connect(sock);
    REVERB_ASSERT_OK(client.status());
    ShmConnection* conn = (*client)->connection();
    // Ring names are constant across rounds (same server epoch + same PID),
    // so reclaim is observable via the unlink of this round's names.
    ShmSegmentNames names = NamesFromConn(*conn);

    std::thread writer_thread([&] { InsertOneItemAsync(client->get()); });
    std::thread sampler_thread([&] { SampleOneItemAsync(client->get()); });
    absl::SleepFor(absl::Milliseconds(100));  // let dispatch enqueue both

    // Crash: the server's HandleDisconnect erases this ClientState while its
    // insert/sample callbacks are still pending on the blocking table.
    ASSERT_GE(conn->control_fd, 0);
    close(conn->control_fd);
    conn->control_fd = -1;  // prevent ~ShmConnection double-close

    ASSERT_TRUE(WaitFor([&] { return !ShmSegmentExists(names.insert_c2s); },
                        absl::Seconds(5)))
        << "round " << round << ": server did not reclaim the crashed client";
    writer_thread.join();
    sampler_thread.join();
    client->reset();
  }

  // Table shutdown fires every accumulated pending callback (kRounds inserts
  // via NotifyPendingInserts, kRounds samples via FinalizeSampleRequest),
  // each dereferencing a ClientState erased rounds ago.
  (*server)->Stop();
  EXPECT_EQ(table->size(), 1);  // only the seeded item
}

// ── Server epoch in segment names (ticket #7) ──

// A second ShmServer on the SAME socket path (crash-restart while the old
// client's segments are still live) must NOT unlink-and-recreate them: the
// old client would keep writing into an orphaned inode nobody reads (silent
// data loss). Segment names carry a per-server epoch so names never collide.
// Observable: fstat nlink on an already-open fd drops to 0 when the segment
// is unlinked out from under us.
TEST(ShmCrashTest, SecondServerSameSocketDoesNotClobberLiveSegments) {
  std::string sock = "/tmp/reverb_shm_crash_" + UniqueTag("epoch") + ".sock";
  auto s1 = ShmServer::Create({MakeTable()}, sock);
  REVERB_ASSERT_OK(s1.status());
  REVERB_ASSERT_OK((*s1)->Start());
  auto c1 = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(c1.status());
  ShmConnection* conn1 = (*c1)->connection();
  const std::string pool1 = conn1->pool_shm_name;
  const std::string ring1 = conn1->insert_c2s.shm_name();

  int pool_fd = shm_open(pool1.c_str(), O_RDWR, 0600);
  ASSERT_GE(pool_fd, 0);
  int ring_fd = shm_open(ring1.c_str(), O_RDWR, 0600);
  ASSERT_GE(ring_fd, 0);

  // Second server on the same socket path; its client has the SAME PID
  // (in-process), so without a server epoch every segment name collides.
  auto s2 = ShmServer::Create({MakeTable()}, sock);
  REVERB_ASSERT_OK(s2.status());
  REVERB_ASSERT_OK((*s2)->Start());
  auto c2 = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(c2.status());

  struct stat st;
  ASSERT_EQ(fstat(pool_fd, &st), 0);
  EXPECT_EQ(st.st_nlink, 1) << "second server unlinked the live pool segment";
  ASSERT_EQ(fstat(ring_fd, &st), 0);
  EXPECT_EQ(st.st_nlink, 1) << "second server unlinked a live client ring";
  EXPECT_NE((*c2)->connection()->pool_shm_name, pool1)
      << "segment names carry no server epoch";
  EXPECT_NE((*c2)->connection()->insert_c2s.shm_name(), ring1)
      << "segment names carry no server epoch";

  close(pool_fd);
  close(ring_fd);
  (*s2)->Stop();
  (*s1)->Stop();
}

// ── Writer ctor member-init race (ticket: shm-writer-worker-race-sigsegv) ──

// Regression for an initialization-order race: the SHM ctor's member-init
// started the RunShmWorker thread while later members (stream_ok_,
// stream_status_) were still UNCONSTRUCTED (members initialize in
// declaration order, and stream_worker_ preceded them). Under CPU
// contention the fresh worker could be scheduled mid-ctor: a garbage
// stream_ok_=false made it skip the data_cv_ wait and copy an unconstructed
// stream_status_, dereferencing a wild StatusRep pointer (SIGSEGV,
// ~3%/suite-run under 10-way parallel load; same signature in Python
// shm_test's concurrent writer test). Fixed by declaring stream_worker_
// LAST in trajectory_writer.h so all members exist before the thread starts
// (ShmSampler never had it: its worker starts in Create(), post-ctor).
//
// The race needs no server traffic: an idle writer's worker evaluates
// stream_ok_ immediately. Hammer create/destroy from several threads to
// maximize the chance a worker gets scheduled mid-ctor. Pre-fix this loop
// segfaults within seconds under load (the whole binary dies — that is the
// regression signal); post-fix it is clean.
TEST(ShmCrashTest, WriterCtorDoesNotRaceMemberInit) {
  auto table = MakeTable();
  std::string sock = "/tmp/reverb_shm_crash_" + UniqueTag("ctor") + ".sock";
  auto server = ShmServer::Create({table}, sock);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());
  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());

  std::atomic<bool> stop{false};
  std::atomic<int64_t> created{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        std::unique_ptr<TrajectoryWriter> writer;
        if (!(*client)
                 ->NewTrajectoryWriter(
                     TrajectoryWriter::Options{
                         .chunker_options =
                             std::make_shared<ConstantChunkerOptions>(1, 1)},
                     &writer)
                 .ok()) {
          return;  // connection trouble; nothing left to hammer
        }
        created.fetch_add(1, std::memory_order_relaxed);
        // ~TrajectoryWriter at scope end: dtor flush + Close + worker join.
      }
    });
  }
  absl::SleepFor(absl::Seconds(5));
  stop.store(true, std::memory_order_relaxed);
  for (auto& th : threads) th.join();
  // Sanity: the loop really hammered (didn't instantly error out).
  EXPECT_GT(created.load(), 100);
  (*server)->Stop();
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
