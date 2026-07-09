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

// Ticket ④: the WRITE direction. A TrajectoryWriter constructed in SHM mode
// (ShmClient::NewTrajectoryWriter) appends data, creates an item, and flushes;
// the item lands in the server's Table over the SHM transport (ALLOCATE +
// INSERT + INSERT_ACK, decision C2/C4). ③'s ShmSampler then reads it back and
// the bytes/dtype/shape must match what was appended. Mirrors
// in_process_client_test.cc's WriteAndSampleRoundTrip, but the writer and the
// sampler talk to a real ShmServer over udsocket + SHM rings + pool.
//
// ponytail: we assert on TensorBuffer bytes/shape/dtype (the transport
// contract), same rationale as shm_sample_test.cc — no CPython interpreter in
// this binary (see shm_sample_test.cc header comment on the libpython/openssl
// clash via :sampler -> grpc).

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/shm/shm_client.h"
#include "reverb/cc/shm/shm_server.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind {
namespace reverb {
namespace shm {
namespace {

using ::testing::SizeIs;

using Step = std::vector<std::optional<TensorBuffer>>;
using StepRef = std::vector<std::optional<std::weak_ptr<CellRef>>>;

const auto kIntSpec = internal::TensorSpec{"0", DataType::Int32, {1}};

// --- Inline tensor builders (no TensorFlow / pybind11 dependency) ---

template <typename T>
std::string RawBytes(const std::vector<int64_t>& shape, T value) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  std::string bytes(static_cast<size_t>(n) * sizeof(T), '\0');
  for (int64_t i = 0; i < n; ++i) {
    std::memcpy(&bytes[static_cast<size_t>(i) * sizeof(T)], &value, sizeof(T));
  }
  return bytes;
}

template <typename T>
TensorBuffer MakeConstantBuffer(DataType dtype_for_spec,
                                const std::vector<int64_t>& shape, T value) {
  return TensorBuffer(TensorSpec{dtype_for_spec, shape},
                      RawBytes<T>(shape, value));
}

template <typename T>
TensorBuffer MakeZeroBuffer(const internal::TensorSpec& spec) {
  return MakeConstantBuffer<T>(spec.dtype, spec.shape, static_cast<T>(0));
}

std::vector<TrajectoryColumn> MakeTrajectory(
    std::vector<std::vector<std::optional<std::weak_ptr<CellRef>>>>
        trajectory) {
  std::vector<TrajectoryColumn> columns;
  for (const auto& optional_refs : trajectory) {
    std::vector<std::weak_ptr<CellRef>> col_refs;
    for (const auto& optional_ref : optional_refs) {
      col_refs.push_back(optional_ref.value());
    }
    columns.push_back(TrajectoryColumn(std::move(col_refs), /*squeeze=*/false));
  }
  return columns;
}

TrajectoryWriter::Options MakeOptions(int max_chunk_length,
                                      int num_keep_alive_refs) {
  return TrajectoryWriter::Options{
      .chunker_options = std::make_shared<ConstantChunkerOptions>(
          max_chunk_length, num_keep_alive_refs),
  };
}

std::shared_ptr<Table> MakeTable(const std::string& name, int max_size = 100) {
  return std::make_shared<Table>(
      /*name=*/name,
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, max_size));
}

// A table whose rate limiter never blocks (min/max_diff wide open): inserts
// and samples proceed freely. Used by the sequential-insert-ack test so the
// ACK path is exercised without the rate limiter gating inserts (which would
// hang the worker's blocking ACK poll when no sampler is draining).
std::shared_ptr<Table> MakePermissiveTable(const std::string& name,
                                            int max_size = 100) {
  return std::make_shared<Table>(
      /*name=*/name,
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/
      std::make_shared<RateLimiter>(1, 1, /*min_diff=*/-1e9, /*max_diff=*/1e9));
}

std::string UniqueTag(const std::string& tag) {
  return tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&tag));
}

// Bring up a server + client for `table` on a fresh udsocket path. The server
// is started; the caller stops it at the end of the test. Returns nullptr on
// failure (asserts are void-returning and can't live in a factory that
// returns a value).
struct ShmFixture {
  std::shared_ptr<Table> table;
  std::unique_ptr<ShmServer> server;
  std::unique_ptr<ShmClient> client;
  std::string sock;

  static std::unique_ptr<ShmFixture> Make(std::shared_ptr<Table> table,
                                          const std::string& tag) {
    auto f = std::make_unique<ShmFixture>();
    f->table = table;
    f->sock = "/tmp/reverb_shm_insert_" + UniqueTag(tag) + ".sock";
    auto s = ShmServer::Create(table, f->sock);
    if (!s.ok()) return nullptr;
    f->server = std::move(*s);
    if (!f->server->Start().ok()) return nullptr;
    auto c = ShmClient::Connect(f->sock);
    if (!c.ok()) return nullptr;
    f->client = std::move(*c);
    return f;
  }
};

void ExpectTensorBufferEqual(const TensorBuffer& x, const TensorBuffer& y) {
  ASSERT_EQ(x.dtype(), y.dtype());
  ASSERT_EQ(x.shape(), y.shape());
  EXPECT_EQ(x.bytes(), y.bytes()) << "byte content differs";
}

// The core ④ test: data written via an SHM TrajectoryWriter lands in the
// server's Table and is read back unchanged through ③'s ShmSampler. Mirrors
// InProcessClientTest.WriteAndSampleRoundTrip.
TEST(ShmInsertTest, WriteAndSampleRoundTrip) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "rt");
  ASSERT_NE(fx, nullptr);

  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(
      fx->client->NewTrajectoryWriter(MakeOptions(1, 1), &writer));

  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush());

  EXPECT_EQ(table->size(), 1);

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(fx->client->NewSampler("t", {1}, &sampler));

  std::vector<TensorBuffer> data;
  REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_THAT(data, SizeIs(1));
  EXPECT_EQ(data[0].dtype(), DataType::Int32);
  // Unsqueezed trajectory column keeps the batch (time) dim -> [1, 1].
  EXPECT_EQ(data[0].shape(), std::vector<int64_t>({1, 1}));

  sampler->Close();
}

// A multi-chunk item: max_chunk_length=2, append 5 steps (chunks of 2,2,1),
// create one item spanning all 5 steps, flush. The writer must serialize each
// ChunkData separately, the server reassemble them, and the sampler return a
// concatenated [5, 1] int32 tensor whose bytes match a direct [5,1] buffer.
TEST(ShmInsertTest, MultiChunkItemRoundTrips) {
  auto table = MakeTable("t");
  auto fx = ShmFixture::Make(table, "mc");
  ASSERT_NE(fx, nullptr);

  std::unique_ptr<TrajectoryWriter> writer;
  // max_chunk_length=2 forces 3 chunks for 5 steps: [2,2,1].
  REVERB_ASSERT_OK(
      fx->client->NewTrajectoryWriter(MakeOptions(2, 5), &writer));

  std::vector<std::weak_ptr<CellRef>> col_refs;
  for (int i = 0; i < 5; i++) {
    StepRef refs;
    REVERB_ASSERT_OK(writer->Append(
        Step({MakeConstantBuffer<int32_t>(DataType::Int32, {1}, i)}), &refs));
    col_refs.push_back(refs[0].value());
  }
  std::vector<TrajectoryColumn> traj;
  traj.push_back(TrajectoryColumn(col_refs, /*squeeze=*/false));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, traj));
  REVERB_ASSERT_OK(writer->Flush());

  EXPECT_EQ(table->size(), 1);

  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(fx->client->NewSampler("t", {1}, &sampler));

  std::vector<TensorBuffer> data;
  REVERB_EXPECT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_THAT(data, SizeIs(1));
  EXPECT_EQ(data[0].shape(), std::vector<int64_t>({5, 1}));
  for (int i = 0; i < 5; i++) {
    const int32_t* p = reinterpret_cast<const int32_t*>(data[0].bytes().data());
    EXPECT_EQ(p[i], i) << "step " << i << " mismatch";
  }

  sampler->Close();
}

// v1 synchronous insert round-trip (decision C2): with a tight rate
// limiter (max_size=1, no samples drained), writing many items must not
// deadlock. v1 RunShmWorker is STRICTLY SYNCHRONOUS (in_flight <= 1): each
// item does ALLOCATE -> INSERT -> read_blocking(ACK) -> erase from
// in_flight -> RELEASE, then loops to the next item. in_flight_items_ never
// exceeds 1, so there is no in_flight>1 pipelined backpressure gate in v1.
// local_can_insert_more_ is set to true on ACK but is NEVER read/awaited by
// RunShmWorker (unlike RunLocalWorker, which waits on it at line ~815) — it is
// vestigial from RunLocalWorker and NOT a backpressure gate in v1. The only
// backpressure here is the natural C->S ring-full block on Write plus the
// serial ACK wait. After draining samples, the remaining items flush through.
//
// ponytail: true in_flight>1 async/pipelined backpressure is deferred —
// upgrade RunShmWorker to async batch + reuse local_can_insert_more_ as the
// gate.
TEST(ShmInsertTest, SequentialInsertAckDoesNotDeadlock) {
  // Permissive rate limiter + small max_size: all 10 inserts complete (the
  // Fifo remover evicts the oldest beyond max_size), exercising the writer's
  // ALLOCATE->INSERT->ACK->RELEASE loop 10 times. If the ACK path wedges,
  // Flush times out instead of hanging.
  auto table = MakePermissiveTable("t", /*max_size=*/2);
  auto fx = ShmFixture::Make(table, "bp");
  ASSERT_NE(fx, nullptr);

  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(
      fx->client->NewTrajectoryWriter(MakeOptions(1, 1), &writer));

  // Write 10 items with a 100ms flush timeout each. If the synchronous
  // INSERT->ACK path is broken (writer never gets confirmation), Flush times
  // out and the test fails with DeadlineExceeded instead of hanging.
  for (int i = 0; i < 10; i++) {
    StepRef refs;
    REVERB_ASSERT_OK(
        writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
    REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  }
  absl::Status st = writer->Flush(/*ignore_last_num_items=*/0,
                                  /*timeout=*/absl::Milliseconds(500));
  REVERB_EXPECT_OK(st) << "Flush timed out — ACK/backpressure path broken";

  writer->Close();
}

// StructuredWriter over SHM: a config with a relative-slice pattern emits
// items into the server's Table as steps are appended. Mirrors
// InProcessClientTest.StructuredWriterConditionWithRelativeSlice.
TEST(ShmInsertTest, StructuredWriterEmitsItemsOverShm) {
  auto table = MakeTable("sw", /*max_size=*/100);
  auto fx = ShmFixture::Make(table, "sw");
  ASSERT_NE(fx, nullptr);

  StructuredWriterConfig config;
  auto* node = config.add_flat();
  node->set_flat_source_index(0);
  node->set_start(-1);  // most recent step
  config.set_table("sw");
  config.mutable_priority()->mutable_constant_fn()->set_value(1.0);
  // Condition: step_index <= 2 (fires on steps 0,1,2).
  auto* cond = config.add_conditions();
  cond->set_step_index(true);
  cond->set_ge(3);
  cond->set_inverse(true);

  std::unique_ptr<StructuredWriter> writer;
  REVERB_ASSERT_OK(fx->client->NewStructuredWriter({config}, &writer));

  for (int i = 0; i < 5; i++) {
    REVERB_ASSERT_OK(writer->Append(
        Step({MakeConstantBuffer<int32_t>(DataType::Int32, {}, i)})));
  }
  REVERB_ASSERT_OK(writer->EndEpisode(/*clear_buffers=*/true));

  // 3 trajectories expected (steps 0,1,2), each a single scalar.
  std::unique_ptr<ShmSampler> sampler;
  REVERB_ASSERT_OK(fx->client->NewSampler("sw", {3}, &sampler));

  std::vector<int32_t> values;
  std::vector<TensorBuffer> data;
  while (sampler->GetNextTrajectory(&data).ok()) {
    ASSERT_EQ(data.size(), 1u);
    const int32_t* p =
        reinterpret_cast<const int32_t*>(data[0].bytes().data());
    values.push_back(*p);
  }
  EXPECT_THAT(values, ::testing::ElementsAre(0, 1, 2));

  sampler->Close();
}

}  // namespace
}  // namespace shm
}  // namespace reverb
}  // namespace deepmind
