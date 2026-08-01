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

// Review #3 (dispatch-thread-hol): the single dispatch thread must never run
// an unbounded synchronous operation. HandleCheckpoint used to execute
// checkpointer_->Save (disk I/O, unbounded) inline on the dispatch thread —
// while it ran, EVERY client's accept/insert/sample response stalled, and
// the client write side spun at 100% CPU (feeding review #2's liveness
// finding). The fix runs Save on a dedicated executor thread and routes the
// response through the client's outbox (the same callback→dispatch-drain
// pattern as async inserts).
//
// This test latches Save inside a test checkpointer, then asserts a second
// client's connect + ServerInfo round-trip still completes promptly. Pre-fix
// the wedged dispatch makes it hang (bounded here by a 5s future timeout);
// post-fix it completes in milliseconds.

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_server.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

// Save signals `entered_` then blocks until `release_` is notified, so the
// test controls exactly how long the (pre-fix: dispatch, post-fix: executor)
// thread is stuck inside the checkpoint.
class LatchedCheckpointer : public Checkpointer {
 public:
  absl::Status Save(std::vector<Table*> tables, int keep_latest,
                    std::string* path) override {
    entered_.Notify();
    release_.WaitForNotification();
    *path = "/tmp/reverb_shm_ckpt_latched";
    return absl::OkStatus();
  }
  absl::Status Load(absl::string_view, ChunkStore*,
                    std::vector<std::shared_ptr<Table>>*) override {
    return absl::UnimplementedError("test double");
  }
  absl::Status LoadLatest(
      std::vector<std::shared_ptr<Table>>*) override {
    return absl::UnimplementedError("test double");
  }
  absl::Status LoadFallbackCheckpoint(
      std::vector<std::shared_ptr<Table>>*) override {
    return absl::UnimplementedError("test double");
  }
  std::string DebugString() const override { return "LatchedCheckpointer"; }

  absl::Notification entered_;
  absl::Notification release_;
};

std::shared_ptr<Table> MakeTable() {
  return std::make_shared<Table>(
      /*name=*/"queue",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/100,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, 100));
}

std::string UniqueTag(const std::string& tag) {
  return tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

TEST(ShmCheckpointTest, CheckpointDoesNotBlockDispatchThread) {
  auto table = MakeTable();
  auto ckpt = std::make_shared<LatchedCheckpointer>();
  std::string sock = "/tmp/reverb_shm_ckpt_" + UniqueTag("hol") + ".sock";
  auto server = ShmServer::Create({table}, sock, ckpt);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client1 = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client1.status());

  // Kick off a checkpoint: pre-fix the dispatch thread runs Save inline and
  // wedges inside it; post-fix Save lands on the checkpoint executor.
  absl::Status ckpt_status;
  std::string ckpt_path;
  std::thread ckpt_thread(
      [&] { ckpt_status = (*client1)->Checkpoint(&ckpt_path); });
  ASSERT_TRUE(
      ckpt->entered_.WaitForNotificationWithTimeout(absl::Seconds(5)))
      << "server never entered Save (checkpoint request not dispatched)";

  // While Save is latched, a SECOND client's connect + ServerInfo must
  // complete promptly. Pre-fix both need the wedged dispatch thread
  // (TryAccept + HandleServerInfo) and hang.
  auto fut = std::async(std::launch::async, [&] {
    auto c2 = ShmClient::Connect(sock);
    if (!c2.ok()) return c2.status();
    std::vector<TableInfo> info;
    return (*c2)->ServerInfo(&info);
  });
  bool completed =
      fut.wait_for(std::chrono::seconds(5)) == std::future_status::ready;

  // Unwind before asserting: Save completes, the wedged (pre-fix) dispatch
  // resumes, client2's connect finishes, the async task exits.
  ckpt->release_.Notify();
  ckpt_thread.join();

  REVERB_EXPECT_OK(ckpt_status);
  EXPECT_NE(ckpt_path, "");
  ASSERT_TRUE(completed)
      << "checkpoint Save blocked the dispatch thread for >5s — "
         "head-of-line blocking (review #3)";
  REVERB_EXPECT_OK(fut.get());

  (*server)->Stop();
}

// The latched checkpoint response must arrive through the outbox→s2c route
// even though the requesting client's insert flow is otherwise idle: the
// dispatch's FlushOutbox drains it after the executor finishes Save. (This
// is what makes ShmClient::Checkpoint's ReadBlocking return post-fix.)
TEST(ShmCheckpointTest, CheckpointResponseRoundTrips) {
  auto table = MakeTable();
  auto ckpt = std::make_shared<LatchedCheckpointer>();
  // Not latched this time: release immediately once entered.
  std::string sock = "/tmp/reverb_shm_ckpt_" + UniqueTag("rt") + ".sock";
  auto server = ShmServer::Create({table}, sock, ckpt);
  REVERB_ASSERT_OK(server.status());
  REVERB_ASSERT_OK((*server)->Start());

  auto client = ShmClient::Connect(sock);
  REVERB_ASSERT_OK(client.status());

  ckpt->release_.Notify();  // Save proceeds as soon as it is entered
  std::string path;
  REVERB_EXPECT_OK((*client)->Checkpoint(&path));
  EXPECT_EQ(path, "/tmp/reverb_shm_ckpt_latched");

  (*server)->Stop();
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
