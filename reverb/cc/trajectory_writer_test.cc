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

#include "reverb/cc/trajectory_writer.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {
namespace {

using Step = std::vector<std::optional<TensorBuffer>>;
using StepRef = std::vector<std::optional<std::weak_ptr<CellRef>>>;

const auto kIntSpec = internal::TensorSpec{"0", DataType::Int32, {1}};
const auto kFloatSpec = internal::TensorSpec{"0", DataType::Float32, {1}};

// --- Inline tensor builders (no TensorFlow dependency) ---

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

inline std::string Int32Name() { return DataTypeName(DataType::Int32); }

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

std::shared_ptr<Table> MakeTable(int max_size = 100) {
  return std::make_shared<Table>(
      /*name=*/"queue",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, max_size));
}

// Wraps a single table into the {name -> Table} map the unchained local
// TrajectoryWriter constructor expects.
internal::flat_hash_map<std::string, std::shared_ptr<Table>> AsMap(
    const std::shared_ptr<Table>& table) {
  return {{table->name(), table}};
}

// ---------------------------------------------------------------------------
// Append validation (exercises the chunker integration without gRPC).
// ---------------------------------------------------------------------------

TEST(TrajectoryWriter, AppendValidatesDtype) {
  auto table = MakeTable();
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/10,
                                             /*num_keep_alive_refs=*/10));
  StepRef refs;

  REVERB_ASSERT_OK(writer.Append(
      Step({MakeZeroBuffer<int32_t>(kIntSpec),
            MakeZeroBuffer<float>(kFloatSpec)}),
      &refs));

  auto status = writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec),
                                    MakeZeroBuffer<int32_t>(kIntSpec)}),
                              &refs);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(absl::StrCat(
                  "Tensor of wrong dtype provided for column 1. Got ",
                  Int32Name(), " but expected Float32.")));
}

TEST(TrajectoryWriter, AppendValidatesShapes) {
  auto table = MakeTable();
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/10,
                                             /*num_keep_alive_refs=*/10));
  StepRef refs;

  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                 &refs));

  auto status =
      writer.Append(Step({MakeConstantBuffer<int32_t>(
                          kIntSpec.dtype, {3}, 0)}),
                    &refs);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Tensor of incompatible shape provided for column 0. "
                  "Got [3] which is incompatible with [1]."));
}

TEST(TrajectoryWriter, AppendAcceptsPartialSteps) {
  auto table = MakeTable();
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/10,
                                             /*num_keep_alive_refs=*/10));

  StepRef both;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeZeroBuffer<int32_t>(kIntSpec),
            MakeZeroBuffer<float>(kFloatSpec)}),
      &both));

  StepRef first_column_only;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeZeroBuffer<int32_t>(kIntSpec), std::nullopt}),
      &first_column_only));
  EXPECT_FALSE(first_column_only[1].has_value());
}

TEST(TrajectoryWriter, AppendPartialRejectsMultipleUsesOfSameColumn) {
  auto table = MakeTable();
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/10,
                                             /*num_keep_alive_refs=*/10));

  StepRef first_column_only;
  REVERB_ASSERT_OK(writer.AppendPartial(
      Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &first_column_only));

  StepRef second_column_only;
  REVERB_ASSERT_OK(writer.AppendPartial(
      Step({std::nullopt, MakeZeroBuffer<float>(kFloatSpec)}),
      &second_column_only));

  StepRef first_column_again;
  auto status = writer.AppendPartial(
      Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &first_column_again);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Append/AppendPartial called with data containing column "
                  "that was present in previous AppendPartial call."));
}

TEST(TrajectoryWriter, EpisodeStepIsIncrementedByAppend) {
  auto table = MakeTable();
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/1,
                                             /*num_keep_alive_refs=*/2));

  EXPECT_EQ(writer.episode_steps(), 0);
  for (int i = 1; i < 11; i++) {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                   &step));
    EXPECT_EQ(writer.episode_steps(), i);
  }

  {
    StepRef step;
    REVERB_ASSERT_OK(writer.AppendPartial(
        Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 10);

  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 11);
}

// ---------------------------------------------------------------------------
// Core local-path round trip: append -> CreateItem -> Flush -> Sample.
// ---------------------------------------------------------------------------

TEST(TrajectoryWriter, LocalRoundTripInsertsAndIsSampleable) {
  auto table = MakeTable(/*max_size=*/10);
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/1,
                                             /*num_keep_alive_refs=*/1));

  // Append a single int32 step.
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                 &refs));

  // Create an item referencing that step and flush it into the table.
  REVERB_ASSERT_OK(
      writer.CreateItem("queue", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // The item must now be in the table and sampleable.
  EXPECT_EQ(table->size(), 1);

  Table::SampledItem sampled;
  REVERB_ASSERT_OK(table->Sample(&sampled));
  ASSERT_TRUE(sampled.ref != nullptr);
  EXPECT_EQ(sampled.ref->table(), "queue");
  EXPECT_EQ(sampled.ref->priority(), 1.0);
  // The trajectory references exactly one chunk.
  ASSERT_EQ(sampled.ref->chunks().size(), 1);
  EXPECT_EQ(sampled.ref->chunks()[0]->num_columns(), 1);
  EXPECT_EQ(sampled.ref->flat_trajectory().columns_size(), 1);
}

TEST(TrajectoryWriter, LocalRoundTripFlushesIncompleteChunk) {
  auto table = MakeTable(/*max_size=*/10);
  // max_chunk_length=2 means the chunk is NOT finalized after one Append.
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/2,
                                             /*num_keep_alive_refs=*/2));

  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                 &refs));

  // The chunk is incomplete; Flush must force-finalize it before inserting.
  EXPECT_FALSE(refs[0]->lock()->IsReady());
  REVERB_ASSERT_OK(
      writer.CreateItem("queue", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer.Flush());
  EXPECT_TRUE(refs[0]->lock()->IsReady());

  EXPECT_EQ(table->size(), 1);
  Table::SampledItem sampled;
  REVERB_ASSERT_OK(table->Sample(&sampled));
  EXPECT_EQ(sampled.ref->chunks().size(), 1);
}

TEST(TrajectoryWriter, LocalRoundTripMultipleItemsShareChunk) {
  auto table = MakeTable(/*max_size=*/10);
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/2,
                                             /*num_keep_alive_refs=*/2));

  // Two steps in the same chunk (max_chunk_length=2).
  StepRef first;
  StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                 &first));
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                 &second));
  // Now the chunk is finalized and both refs share the same chunk_key.
  ASSERT_TRUE(first[0]->lock()->IsReady());
  ASSERT_TRUE(second[0]->lock()->IsReady());
  EXPECT_EQ(first[0]->lock()->chunk_key(), second[0]->lock()->chunk_key());

  // Two items referencing the same chunk.
  REVERB_ASSERT_OK(
      writer.CreateItem("queue", 1.0, MakeTrajectory({{first[0]}})));
  REVERB_ASSERT_OK(
      writer.CreateItem("queue", 2.0, MakeTrajectory({{second[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  EXPECT_EQ(table->size(), 2);

  // Sample both items; both should reference the single shared chunk.
  std::vector<uint64_t> chunk_keys;
  for (int i = 0; i < 2; ++i) {
    Table::SampledItem sampled;
    REVERB_ASSERT_OK(table->Sample(&sampled));
    ASSERT_EQ(sampled.ref->chunks().size(), 1);
    chunk_keys.push_back(sampled.ref->chunks()[0]->key());
  }
  EXPECT_EQ(chunk_keys[0], chunk_keys[1]);
}

TEST(TrajectoryWriter, LocalRoundTripEndEpisodeFinalizesChunks) {
  auto table = MakeTable(/*max_size=*/10);
  TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/2,
                                             /*num_keep_alive_refs=*/2));

  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                 &step));
  EXPECT_FALSE(step[0]->lock()->IsReady());

  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));
  EXPECT_TRUE(step[0]->lock()->IsReady());
  EXPECT_EQ(writer.episode_steps(), 0);
}

TEST(TrajectoryWriter, LocalRoundTripDestructorFlushesPending) {
  auto table = MakeTable(/*max_size=*/10);
  {
    TrajectoryWriter writer(AsMap(table), MakeOptions(/*max_chunk_length=*/2,
                                               /*num_keep_alive_refs=*/2));
    StepRef refs;
    REVERB_ASSERT_OK(writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}),
                                   &refs));
    REVERB_ASSERT_OK(
        writer.CreateItem("queue", 1.0, MakeTrajectory({{refs[0]}})));
    // No explicit Flush/EndEpisode; the destructor must flush the item.
  }
  EXPECT_EQ(table->size(), 1);
}

// ---------------------------------------------------------------------------
// Multi-table local path: writer holds a {name -> Table} map and dispatches
// by item.table(). (Unchained from the single-table_ binding.)
// ---------------------------------------------------------------------------

// Builds a {name -> Table} map with the given (name, max_size) pairs. Each
// table uses a Queue(1) limiter so inserts proceed without sampling pressure.
internal::flat_hash_map<std::string, std::shared_ptr<Table>> MakeTables(
    std::vector<std::pair<std::string, int>> named_sizes) {
  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables;
  for (auto& [name, max_size] : named_sizes) {
    tables[name] = std::make_shared<Table>(
        /*name=*/name,
        /*sampler=*/std::make_shared<FifoSelector>(),
        /*remover=*/std::make_shared<FifoSelector>(),
        /*max_size=*/max_size,
        /*max_times_sampled=*/1,
        /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, max_size));
  }
  return tables;
}

TEST(TrajectoryWriter, LocalRoundTripDispatchesToMultipleTables) {
  auto tables = MakeTables({{"a", 10}, {"b", 10}});
  TrajectoryWriter writer(tables, MakeOptions(/*max_chunk_length=*/1,
                                             /*num_keep_alive_refs=*/1));

  StepRef refs_a;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs_a));
  REVERB_ASSERT_OK(writer.CreateItem("a", 1.0, MakeTrajectory({{refs_a[0]}})));

  StepRef refs_b;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs_b));
  REVERB_ASSERT_OK(writer.CreateItem("b", 2.0, MakeTrajectory({{refs_b[0]}})));

  REVERB_ASSERT_OK(writer.Flush());

  EXPECT_EQ(tables["a"]->size(), 1);
  EXPECT_EQ(tables["b"]->size(), 1);

  Table::SampledItem sampled_a;
  REVERB_ASSERT_OK(tables["a"]->Sample(&sampled_a));
  EXPECT_EQ(sampled_a.ref->table(), "a");
  EXPECT_EQ(sampled_a.ref->priority(), 1.0);

  Table::SampledItem sampled_b;
  REVERB_ASSERT_OK(tables["b"]->Sample(&sampled_b));
  EXPECT_EQ(sampled_b.ref->table(), "b");
  EXPECT_EQ(sampled_b.ref->priority(), 2.0);
}

TEST(TrajectoryWriter, LocalRoundTripRejectsCreateItemForUnknownTable) {
  auto tables = MakeTables({{"a", 10}, {"b", 10}});
  TrajectoryWriter writer(tables, MakeOptions(/*max_chunk_length=*/1,
                                             /*num_keep_alive_refs=*/1));

  StepRef refs;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  // CreateItem itself succeeds (validation is deferred); the error surfaces
  // when the worker tries to dispatch to a table not in the map.
  REVERB_ASSERT_OK(writer.CreateItem("c", 1.0, MakeTrajectory({{refs[0]}})));

  // Flush propagates the worker's kNotFound error as the unrecoverable status.
  auto status = writer.Flush(/*ignore_last_num_items=*/0,
                             /*timeout=*/absl::Milliseconds(500));
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
}

TEST(TrajectoryWriter, LocalRoundTripBackpressureAcrossTables) {
  // max_size=1 => max_enqueued_inserts=1. A Queue(1) limiter blocks inserts
  // until a sample is drawn, so the second insert cannot be queued and the
  // writer-level backpressure engages (can_insert_more=false).
  auto tables = MakeTables({{"a", 1}});
  TrajectoryWriter writer(tables, MakeOptions(/*max_chunk_length=*/1,
                                             /*num_keep_alive_refs=*/4));

  for (int i = 0; i < 4; ++i) {
    StepRef refs;
    REVERB_ASSERT_OK(
        writer.Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
    REVERB_ASSERT_OK(writer.CreateItem("a", 1.0, MakeTrajectory({{refs[0]}})));
  }

  // With the table blocked (no samples drawn) the worker cannot drain all
  // items, so Flush times out. This is the writer-level backpressure engaging.
  auto status = writer.Flush(/*ignore_last_num_items=*/0,
                             /*timeout=*/absl::Milliseconds(100));
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);

  // Draining one sample lets one insert complete, but the remaining items keep
  // the writer blocked (still writer-level backpressure, single channel).
  Table::SampledItem sampled;
  REVERB_ASSERT_OK(tables["a"]->Sample(&sampled));
  // Still blocked on the rest.
  auto status2 = writer.Flush(/*ignore_last_num_items=*/0,
                              /*timeout=*/absl::Milliseconds(100));
  EXPECT_EQ(status2.code(), absl::StatusCode::kDeadlineExceeded);

  writer.Close();
}

// ---------------------------------------------------------------------------
// Options validation.
// ---------------------------------------------------------------------------

TEST(TrajectoryWriterOptionsTest, Valid) {
  TrajectoryWriter::Options options =
      MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2);
  REVERB_EXPECT_OK(options.Validate());
}

TEST(TrajectoryWriterOptionsTest, NoChunkerOptions) {
  TrajectoryWriter::Options options;
  auto status = options.Validate();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("chunker_options must be set."));
}

TEST(TrajectoryWriterOptionsTest, ZeroMaxChunkLength) {
  auto options = MakeOptions(/*max_chunk_length=*/0, /*num_keep_alive_refs=*/2);
  auto status = options.Validate();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("max_chunk_length must be > 0 but got 0."));
}

TEST(TrajectoryWriterOptionsTest, NumKeepAliveLtMaxChunkLength) {
  auto options = MakeOptions(/*max_chunk_length=*/6, /*num_keep_alive_refs=*/5);
  auto status = options.Validate();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "num_keep_alive_refs (5) must be >= max_chunk_length (6)."));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
