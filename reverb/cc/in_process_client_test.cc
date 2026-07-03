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

#include "reverb/cc/in_process_client.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind {
namespace reverb {
namespace {

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

// End-to-end: write a trajectory via InProcessClient -> TrajectoryWriter,
// sample it back via InProcessClient -> Sampler, validate round-trip.
TEST(InProcessClientTest, WriteAndSampleRoundTrip) {
  auto table = MakeTable("t");
  InProcessClient client({table});

  // Write one int32 step and create/flush an item into the table.
  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(
      client.NewTrajectoryWriter("t", MakeOptions(1, 1), &writer));

  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush());

  EXPECT_EQ(table->size(), 1);

  // Sample it back through the client-created Sampler.
  Sampler::Options sopts;
  sopts.max_samples = 1;
  std::unique_ptr<Sampler> sampler;
  REVERB_ASSERT_OK(client.NewSampler("t", sopts, &sampler));

  std::vector<TensorBuffer> data;
  REVERB_ASSERT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_EQ(data.size(), 1u);
  EXPECT_EQ(data[0].dtype(), DataType::Int32);
  // Unsqueezed trajectory column keeps the batch (time) dim -> [1, 1].
  EXPECT_EQ(data[0].shape(), std::vector<int64_t>({1, 1}));
}

TEST(InProcessClientTest, NewTrajectoryWriterRejectsUnknownTable) {
  auto table = MakeTable("t");
  InProcessClient client({table});
  std::unique_ptr<TrajectoryWriter> writer;
  auto status = client.NewTrajectoryWriter("missing", MakeOptions(1, 1), &writer);
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
}

TEST(InProcessClientTest, NewSamplerRejectsUnknownTable) {
  auto table = MakeTable("t");
  InProcessClient client({table});
  std::unique_ptr<Sampler> sampler;
  auto status = client.NewSampler("missing", Sampler::Options(), &sampler);
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
}

TEST(InProcessClientTest, MutatePrioritiesAndReset) {
  auto table = MakeTable("t");
  InProcessClient client({table});

  // Insert an item the lazy way: borrow a writer.
  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(
      client.NewTrajectoryWriter("t", MakeOptions(1, 1), &writer));
  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush());
  ASSERT_EQ(table->size(), 1);

  // Grab the single item's key and update its priority.
  auto items = table->Copy();
  ASSERT_EQ(items.size(), 1u);
  uint64_t key = items[0].key();
  KeyWithPriority update;
  update.set_key(key);
  update.set_priority(5.0);
  REVERB_ASSERT_OK(client.MutatePriorities("t", {update}, {}));
  EXPECT_EQ(table->Copy()[0].priority(), 5.0);

  // Delete via MutatePriorities deletes list.
  REVERB_ASSERT_OK(client.MutatePriorities("t", {}, {key}));
  EXPECT_EQ(table->size(), 0);

  // Re-insert then Reset clears the table.
  StepRef refs2;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs2));
  REVERB_ASSERT_OK(writer->CreateItem("t", 1.0, MakeTrajectory({{refs2[0]}})));
  REVERB_ASSERT_OK(writer->Flush());
  EXPECT_EQ(table->size(), 1);
  REVERB_ASSERT_OK(client.Reset("t"));
  EXPECT_EQ(table->size(), 0);
}

TEST(InProcessClientTest, ServerInfoAggregatesAllTables) {
  auto t1 = MakeTable("t1");
  auto t2 = MakeTable("t2");
  InProcessClient client({t1, t2});

  std::vector<TableInfo> info;
  REVERB_ASSERT_OK(client.ServerInfo(&info));
  ASSERT_EQ(info.size(), 2u);
  std::set<std::string> names = {info[0].name(), info[1].name()};
  EXPECT_EQ(names, (std::set<std::string>{"t1", "t2"}));
}

TEST(InProcessClientTest, CheckpointWithoutCheckpointerFails) {
  auto table = MakeTable("t");
  InProcessClient client({table});  // no checkpointer
  std::string path;
  auto status = client.Checkpoint(&path);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
