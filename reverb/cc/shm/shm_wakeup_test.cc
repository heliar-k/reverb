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

// Ticket 03: the dispatch thread blocks in poll() when quiescent instead of
// spinning on sched_yield()+usleep(50), and is woken in ms by (a) a client
// WriteBlocking that found its ring's shared `server_asleep` flag set, or
// (b) WakeDispatch() from an off-dispatch producer / Stop(). These tests
// cover: idle poll cadence (no spin), ms-level wakeup on the insert and
// sample flows, clean Stop() out of a blocked poll, and near-zero idle CPU.
//
// ponytail: links libpython via :shm_client/:shm_server -> :tensor_proxy,
// same as shm_insert_test (see its header comment).

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

#include "gtest/gtest.h"

#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_server.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

const auto kIntSpec = TensorSpec{DataType::Int32, {1}};

TensorBuffer MakeZeroBuffer() {
  std::string bytes(sizeof(int32_t), '\0');
  return TensorBuffer(kIntSpec, bytes);
}

std::shared_ptr<Table> MakeTable(const std::string& name, int max_size = 100) {
  return std::make_shared<Table>(
      name, std::make_shared<FifoSelector>(), std::make_shared<FifoSelector>(),
      max_size, /*max_times_sampled=*/1,
      std::make_shared<RateLimiter>(1, 1, 0, max_size));
}

std::string UniqueTag(const std::string& tag) {
  return tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

struct ShmFixture {
  std::shared_ptr<Table> table;
  std::unique_ptr<ShmServer> server;
  std::unique_ptr<ShmClient> client;

  static std::unique_ptr<ShmFixture> Make(std::shared_ptr<Table> table,
                                          const std::string& tag) {
    auto f = std::make_unique<ShmFixture>();
    f->table = table;
    std::string sock = "/tmp/reverb_shm_wakeup_" + UniqueTag(tag) + ".sock";
    auto s = ShmServer::Create({table}, sock, nullptr);
    if (!s.ok()) return nullptr;
    f->server = std::move(*s);
    if (!f->server->Start().ok()) return nullptr;
    auto c = ShmClient::Connect(sock);
    if (!c.ok()) return nullptr;
    f->client = std::move(*c);
    return f;
  }
};

// Poll `cond` (100us granularity) until it holds or `timeout` elapses.
bool WaitFor(const std::function<bool()>& cond, absl::Duration timeout) {
  absl::Time deadline = absl::Now() + timeout;
  while (absl::Now() < deadline) {
    if (cond()) return true;
    absl::SleepFor(absl::Microseconds(100));
  }
  return cond();
}

// Waits until the dispatch thread has flagged itself asleep on `ring` (the
// flag lives in shared memory, so the client observes it directly). Failing
// this means the server never blocked in poll() at all.
void AssertServerAsleep(Ring* ring) {
  ASSERT_TRUE(WaitFor(
      [ring] {
        return ring->server_asleep()->load(std::memory_order_seq_cst) != 0;
      },
      absl::Seconds(3)))
      << "dispatch thread never entered its blocking poll";
}

// An idle server must sit in the 50ms-fallback poll (~20 entries/s), not
// spin. Zero clients: the poll set is just listen_fd + wake_fd_.
TEST(ShmWakeupTest, IdleServerBlocksInPollWithoutSpinning) {
  auto fx = ShmFixture::Make(MakeTable("t"), "idle");
  ASSERT_NE(fx, nullptr);

  uint64_t before = fx->server->poll_entries_for_test();
  absl::SleepFor(absl::Seconds(1));
  uint64_t delta = fx->server->poll_entries_for_test() - before;

  EXPECT_GE(delta, 3u) << "dispatch never blocked in poll (or never idled)";
  EXPECT_LE(delta, 45u) << "dispatch is falling out of poll far above the "
                           "50ms fallback rate (~20/s) — spurious wakeups";
}

// Idle CPU: with a client connected but no traffic, whole-process CPU
// (utime+stime, which includes the dispatch thread) stays ~0. Pre-ticket-03
// the dispatch loop alone burned ~20k passes/s. Bound is deliberately loose
// (250ms over 2s) for loaded CI runners.
TEST(ShmWakeupTest, IdleServerConsumesNoCpu) {
  auto fx = ShmFixture::Make(MakeTable("t"), "cpu");
  ASSERT_NE(fx, nullptr);
  // Let the post-connect burst (welcome/server_info) settle and the dispatch
  // thread go to sleep first.
  AssertServerAsleep(&fx->client->connection()->insert_c2s);

  auto cpu_now = [] {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return absl::Seconds(ru.ru_utime.tv_sec) +
           absl::Microseconds(ru.ru_utime.tv_usec) +
           absl::Seconds(ru.ru_stime.tv_sec) +
           absl::Microseconds(ru.ru_stime.tv_usec);
  };
  absl::Duration start = cpu_now();
  absl::SleepFor(absl::Seconds(2));
  absl::Duration used = cpu_now() - start;

  EXPECT_LT(used, absl::Milliseconds(250)) << "idle server burned CPU";
}

// Wakeup correctness, insert flow: with the dispatch thread asleep, a
// client request that rides WriteBlocking (SERVER_INFO round-trip, the
// lightest insert-flow op) must be answered in ms — not at the next 50ms
// fallback tick.
TEST(ShmWakeupTest, SleepingServerAnswersInsertFlowRequestFast) {
  auto fx = ShmFixture::Make(MakeTable("t"), "wakeins");
  ASSERT_NE(fx, nullptr);
  AssertServerAsleep(&fx->client->connection()->insert_c2s);

  absl::Time t0 = absl::Now();
  std::vector<TableInfo> info;
  REVERB_ASSERT_OK(fx->client->ServerInfo(&info));
  absl::Duration elapsed = absl::Now() - t0;

  EXPECT_EQ(info.size(), 1u);
  EXPECT_LT(elapsed, absl::Milliseconds(250))
      << "asleep server took " << elapsed
      << " to answer — the wake byte did not arrive";
}

// Wakeup correctness, sample flow: an item inserted while the server is
// awake; the sampler's SAMPLE request then lands while dispatch is asleep
// and must still get its SAMPLE_RESP in ms (this exercises the sample_c2s
// asleep flag and the sample-completion WakeDispatch).
TEST(ShmWakeupTest, SleepingServerAnswersSampleFast) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "wakesmp");
  ASSERT_NE(fx, nullptr);

  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(fx->client->NewTrajectoryWriter(
      TrajectoryWriter::Options{
          .chunker_options = std::make_shared<ConstantChunkerOptions>(1, 1)},
      &writer));
  std::vector<absl::optional<std::weak_ptr<CellRef>>> refs;
  REVERB_ASSERT_OK(
      writer->Append(std::vector<absl::optional<TensorBuffer>>{MakeZeroBuffer()},
                     &refs));
  std::vector<std::weak_ptr<CellRef>> col{refs[0].value()};
  REVERB_ASSERT_OK(writer->CreateItem(
      "t", 1.0, std::vector<TrajectoryColumn>{TrajectoryColumn(col, false)}));
  REVERB_ASSERT_OK(writer->Flush());
  ASSERT_EQ(table->size(), 1);
  writer->Close();

  AssertServerAsleep(&fx->client->connection()->sample_c2s);

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(fx->client->NewSampler("t", {1}, &sampler));
  absl::Time t0 = absl::Now();
  std::vector<TensorBuffer> data;
  REVERB_ASSERT_OK(sampler->GetNextTrajectory(&data));
  absl::Duration elapsed = absl::Now() - t0;

  EXPECT_EQ(data.size(), 1u);
  EXPECT_LT(elapsed, absl::Milliseconds(250))
      << "asleep server took " << elapsed << " to answer a SAMPLE";
  sampler->Close();
}

// Stop() must pull the dispatch thread out of a blocked poll immediately
// (running_=false + WakeDispatch), not wait out a fallback tick or hang.
TEST(ShmWakeupTest, StopExitsBlockedPollCleanly) {
  auto table = MakeTable("t");
  std::string sock = "/tmp/reverb_shm_wakeup_stop_" + UniqueTag("s") + ".sock";
  auto s = ShmServer::Create({table}, sock, nullptr);
  REVERB_ASSERT_OK(s.status());
  auto server = std::move(*s);
  REVERB_ASSERT_OK(server->Start());
  ASSERT_TRUE(WaitFor(
      [&] { return server->poll_entries_for_test() > 0; },
      absl::Seconds(3)))
      << "dispatch never blocked in poll";

  absl::Time t0 = absl::Now();
  server->Stop();
  EXPECT_LT(absl::Now() - t0, absl::Seconds(1)) << "Stop() hung on the poll";
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
