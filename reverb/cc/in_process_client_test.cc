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
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/default/simple_checkpointer.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/structured_writer.h"
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
      client.NewTrajectoryWriter(MakeOptions(1, 1), &writer));

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
  // After unchaining, NewTrajectoryWriter takes no `table` argument (the
  // writer holds all client tables and dispatches by item.table()). An
  // unknown table is rejected at CreateItem time by ItemAndRefs::Validate
  // (the flat_signature_map built from the client's tables doesn't contain
  // it), mirroring the gRPC path's deferred validation.
  auto table = MakeTable("t");
  InProcessClient client({table});
  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(client.NewTrajectoryWriter(MakeOptions(1, 1), &writer));
  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  auto status = writer->CreateItem("missing", 1.0, MakeTrajectory({{refs[0]}}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("could not be found"));
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
      client.NewTrajectoryWriter(MakeOptions(1, 1), &writer));
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

// ponytail: 临时 checkpoint 根目录用 mkdtemp 生成唯一名,避免跨测试冲突。
static std::string MakeCkptRoot() {
  auto base = std::filesystem::temp_directory_path() / "inproc_ckptXXXXXX";
  std::string tmpl = base.string();
  char* dir = mkdtemp(tmpl.data());
  REVERB_CHECK(dir != nullptr) << "mkdtemp failed for " << tmpl;
  return dir;
}

TEST(InProcessClientTest, CheckpointSaveAndLoadLatestRoundTrip) {
  // Save 阶段:写一个 item,checkpoint。
  auto table = MakeTable("ckpt_table");
  auto checkpointer =
      std::make_shared<SimpleCheckpointer>(MakeCkptRoot());

  std::vector<std::shared_ptr<Table>> save_tables{table};
  InProcessClient saver(save_tables, checkpointer);

  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(
      saver.NewTrajectoryWriter(MakeOptions(1, 1), &writer));
  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(
      writer->CreateItem("ckpt_table", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush());
  ASSERT_EQ(table->size(), 1);

  std::string path;
  REVERB_ASSERT_OK(saver.Checkpoint(&path));
  ASSERT_TRUE(std::filesystem::exists(path));

  // LoadLatest 阶段:用一个 *同 name* 的空 table 的新 client 从同 checkpointer 恢复。
  auto loaded_table = MakeTable("ckpt_table");
  std::vector<std::shared_ptr<Table>> load_tables{loaded_table};
  InProcessClient loader(load_tables, checkpointer);
  REVERB_ASSERT_OK(loader.LoadLatest());
  // item 数量恢复为 1。
  EXPECT_EQ(loaded_table->size(), 1);

  // 采样 round-trip:加载的 item 能正常采样出原数据。
  Sampler::Options sopts;
  sopts.max_samples = 1;
  std::unique_ptr<Sampler> sampler;
  REVERB_ASSERT_OK(loader.NewSampler("ckpt_table", sopts, &sampler));
  std::vector<TensorBuffer> data;
  REVERB_ASSERT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_EQ(data.size(), 1u);
  EXPECT_EQ(data[0].dtype(), DataType::Int32);
  EXPECT_EQ(data[0].shape(), std::vector<int64_t>({1, 1}));
}

TEST(InProcessClientTest, LoadLatestWithoutCheckpointerFails) {
  auto table = MakeTable("t");
  InProcessClient client({table});  // no checkpointer
  EXPECT_EQ(client.LoadLatest().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(InProcessClientTest, LoadLatestOnEmptyRootReturnsNotFound) {
  auto table = MakeTable("t");
  auto checkpointer =
      std::make_shared<SimpleCheckpointer>(MakeCkptRoot());
  InProcessClient client({table}, checkpointer);
  // 空根目录无 checkpoint -> NotFound(首次启动正常)。
  EXPECT_EQ(client.LoadLatest().code(), absl::StatusCode::kNotFound);
}

TEST(InProcessClientTest, LoadFromPathRoundTrip) {
  // Save 阶段:写一个 item,checkpoint 到 path。
  auto table = MakeTable("ckpt_table");
  auto checkpointer =
      std::make_shared<SimpleCheckpointer>(MakeCkptRoot());

  std::vector<std::shared_ptr<Table>> save_tables{table};
  InProcessClient saver(save_tables, checkpointer);

  std::unique_ptr<TrajectoryWriter> writer;
  REVERB_ASSERT_OK(
      saver.NewTrajectoryWriter(MakeOptions(1, 1), &writer));
  StepRef refs;
  REVERB_ASSERT_OK(
      writer->Append(Step({MakeZeroBuffer<int32_t>(kIntSpec)}), &refs));
  REVERB_ASSERT_OK(
      writer->CreateItem("ckpt_table", 1.0, MakeTrajectory({{refs[0]}})));
  REVERB_ASSERT_OK(writer->Flush());
  ASSERT_EQ(table->size(), 1);

  std::string path;
  REVERB_ASSERT_OK(saver.Checkpoint(&path));
  ASSERT_TRUE(std::filesystem::exists(path));

  // Load(path) 阶段:用一个 *同 name* 的空 table 的新 client 从指定 path
  // 恢复。Load 用 loader 自有的 chunk_store_ 接收 checkpoint 中的 chunk。
  auto loaded_table = MakeTable("ckpt_table");
  std::vector<std::shared_ptr<Table>> load_tables{loaded_table};
  InProcessClient loader(load_tables, checkpointer);
  REVERB_ASSERT_OK(loader.Load(path));
  EXPECT_EQ(loaded_table->size(), 1);

  // 采样 round-trip:加载的 item 能正常采样出原数据。
  Sampler::Options sopts;
  sopts.max_samples = 1;
  std::unique_ptr<Sampler> sampler;
  REVERB_ASSERT_OK(loader.NewSampler("ckpt_table", sopts, &sampler));
  std::vector<TensorBuffer> data;
  REVERB_ASSERT_OK(sampler->GetNextTrajectory(&data));
  ASSERT_EQ(data.size(), 1u);
  EXPECT_EQ(data[0].dtype(), DataType::Int32);
  EXPECT_EQ(data[0].shape(), std::vector<int64_t>({1, 1}));
}

TEST(InProcessClientTest, LoadWithoutCheckpointerFails) {
  auto table = MakeTable("t");
  InProcessClient client({table});  // no checkpointer
  EXPECT_EQ(client.Load("/any/path").code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(InProcessClientTest, MutatePrioritiesUnknownTableFails) {
  auto table = MakeTable("t");
  InProcessClient client({table});
  EXPECT_EQ(client.MutatePriorities("missing", {}, {}).code(),
            absl::StatusCode::kNotFound);
}

TEST(InProcessClientTest, ResetUnknownTableFails) {
  auto table = MakeTable("t");
  InProcessClient client({table});
  EXPECT_EQ(client.Reset("missing").code(), absl::StatusCode::kNotFound);
}

TEST(InProcessClientTest, NewStructuredWriterRejectsEmptyConfigs) {
  auto table = MakeTable("t");
  InProcessClient client({table});
  std::unique_ptr<StructuredWriter> writer;
  EXPECT_EQ(client.NewStructuredWriter({}, &writer).code(),
            absl::StatusCode::kInvalidArgument);
}

// REVERB_CHECK 无 NDEBUG 防护:opt/dbg 下均评估并 abort,死亡测试稳定触发。
TEST(InProcessClientTest, ConstructorRejectsNullTable) {
  ASSERT_DEATH(InProcessClient client({nullptr}), "null table");
}

// Regression for A9: StructuredWriter in in-process mode with a condition
// that fires multiple times AND a relative slice (start=-1) must yield one
// trajectory per matching step, each carrying that step's value.
//
// Drives the real TrajectoryWriter(table) in-process path via
// InProcessClient::NewStructuredWriter (the C++ FakeWriter in
// structured_writer_test.cc does NOT exercise this path). MakeTable defaults
// to max_times_sampled=1, so each sampled item is removed after one draw;
// this matches the gRPC Table.queue semantics used by the gRPC
// test_single_condition. With max_times_sampled=0 the Fifo sampler would
// repeatedly return the same (oldest) item, which is correct sampler
// behaviour, not an engine bug.
TEST(InProcessClientTest, StructuredWriterConditionWithRelativeSlice) {
  auto table = MakeTable("sw", /*max_size=*/100);
  InProcessClient client({table});

  // Pattern: x[-1] -> PatternNode(flat_source_index=0, start=-1, stop unset).
  StructuredWriterConfig config;
  auto* node = config.add_flat();
  node->set_flat_source_index(0);
  node->set_start(-1);  // relative to most recent step
  config.set_table("sw");
  config.mutable_priority()->mutable_constant_fn()->set_value(1.0);
  // Condition: step_index <= 2  ==  NOT(step_index >= 3), fires on steps 0,1,2.
  // (The Python Condition encodes <= via inverse ge, see structured_writer.py.)
  auto* cond = config.add_conditions();
  cond->set_step_index(true);
  cond->set_ge(3);
  cond->set_inverse(true);

  std::unique_ptr<StructuredWriter> writer;
  REVERB_ASSERT_OK(client.NewStructuredWriter({config}, &writer));

  for (int i = 0; i < 5; i++) {
    // Python `writer.append(i)` passes a Python int, which becomes a 0-d
    // (scalar) ndarray via np.asarray. Mirror that here with a scalar shape.
    REVERB_ASSERT_OK(writer->Append(
        Step({MakeConstantBuffer<int32_t>(DataType::Int32, {}, i)})));
  }
  REVERB_ASSERT_OK(writer->EndEpisode(/*clear_buffers=*/true));

  // Drain the table: 3 trajectories expected, each a single scalar.
  Sampler::Options sopts;
  sopts.max_samples = 3;
  std::unique_ptr<Sampler> sampler;
  REVERB_ASSERT_OK(client.NewSampler("sw", sopts, &sampler));

  std::vector<int32_t> values;
  std::vector<TensorBuffer> data;
  while (sampler->GetNextTrajectory(&data).ok()) {
    ASSERT_EQ(data.size(), 1u);
    // squeezed single-row column -> scalar shape {}
    const int32_t* p =
        reinterpret_cast<const int32_t*>(data[0].bytes().data());
    values.push_back(*p);
  }

  // gRPC path (Table.queue, max_times_sampled=1) yields {0, 1, 2}. The
  // in-process engine produces the same three distinct items; the earlier
  // "[0,0,0]" report was caused by sampling with max_times_sampled=0, not by
  // an engine defect.
  EXPECT_THAT(values, ::testing::ElementsAre(0, 1, 2));
}

// D1/D3: StructuredWriter dispatches to multiple tables via config.table().
// After unchaining, NewStructuredWriter takes no `table` argument; each
// config's `table` field routes the item to the right table through the
// underlying TrajectoryWriter's `tables_` map.
TEST(InProcessClientTest, StructuredWriterDispatchesToMultipleTablesViaConfigs) {
  auto table_a = MakeTable("a", /*max_size=*/100);
  auto table_b = MakeTable("b", /*max_size=*/100);
  InProcessClient client({table_a, table_b});

  // Config A: emit the latest step into table "a".
  StructuredWriterConfig config_a;
  auto* node_a = config_a.add_flat();
  node_a->set_flat_source_index(0);
  node_a->set_start(-1);  // most recent step
  config_a.set_table("a");
  config_a.mutable_priority()->mutable_constant_fn()->set_value(1.0);

  // Config B: same pattern, different table.
  StructuredWriterConfig config_b;
  auto* node_b = config_b.add_flat();
  node_b->set_flat_source_index(0);
  node_b->set_start(-1);
  config_b.set_table("b");
  config_b.mutable_priority()->mutable_constant_fn()->set_value(2.0);

  std::unique_ptr<StructuredWriter> writer;
  REVERB_ASSERT_OK(client.NewStructuredWriter({config_a, config_b}, &writer));

  REVERB_ASSERT_OK(writer->Append(
      Step({MakeConstantBuffer<int32_t>(DataType::Int32, {}, 7)})));
  REVERB_ASSERT_OK(writer->EndEpisode(/*clear_buffers=*/true));

  EXPECT_EQ(table_a->size(), 1);
  EXPECT_EQ(table_b->size(), 1);

  // Each table received an item carrying the appended value, with its own
  // priority.
  Sampler::Options sopts;
  sopts.max_samples = 1;

  std::unique_ptr<Sampler> sampler_a;
  REVERB_ASSERT_OK(client.NewSampler("a", sopts, &sampler_a));
  std::vector<TensorBuffer> data_a;
  REVERB_ASSERT_OK(sampler_a->GetNextTrajectory(&data_a));
  ASSERT_EQ(data_a.size(), 1u);
  EXPECT_EQ(*reinterpret_cast<const int32_t*>(data_a[0].bytes().data()), 7);

  std::unique_ptr<Sampler> sampler_b;
  REVERB_ASSERT_OK(client.NewSampler("b", sopts, &sampler_b));
  std::vector<TensorBuffer> data_b;
  REVERB_ASSERT_OK(sampler_b->GetNextTrajectory(&data_b));
  ASSERT_EQ(data_b.size(), 1u);
  EXPECT_EQ(*reinterpret_cast<const int32_t*>(data_b[0].bytes().data()), 7);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
