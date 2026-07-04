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

#include "reverb/cc/platform/default/simple_checkpointer.h"

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/table.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace {

// ponytail: 原 tfrecord_checkpointer_test 用 proto_test_util::MakeChunkData,
// 但那依赖 TF(tensorflow::Tensor + CompressTensorAsProto)。本测试是原生 cc_test
// 不注入 TF,而 SimpleCheckpointer 路径不解码 tensor 字节(Table/ChunkStore 只读
// chunk_key / sequence_range / data_tensors_len),故手搓一个最小 ChunkData:
// 一个 DT_INT32 tensor,shape [1],4 字节 content。够 round-trip。

ChunkData MakeChunkData(uint64_t key) {
  ChunkData chunk;
  chunk.set_chunk_key(key);
  auto* range = chunk.mutable_sequence_range();
  range->set_episode_id(key * 100);
  range->set_start(0);
  range->set_end(0);  // num_rows = end - start + 1 = 1
  auto* tensor = chunk.mutable_data()->add_tensors();
  tensor->set_dtype(::reverb::tensor::DT_INT32);
  tensor->mutable_shape()->add_dim(1);  // 1 row
  tensor->set_tensor_content(std::string(4, '\1'));  // 1 个 int32
  chunk.set_data_tensors_len(1);
  return chunk;
}

// 构造一个引用 `chunks` 的 PrioritizedItem:单列、单 slice、覆盖整 chunk。
PrioritizedItem MakePrioritizedItem(const std::string& table, uint64_t key,
                                    double priority,
                                    const std::vector<ChunkData>& chunks) {
  PrioritizedItem item;
  item.set_key(key);
  item.set_table(table);
  item.set_priority(priority);
  auto* col = item.mutable_flat_trajectory()->add_columns();
  for (const auto& chunk : chunks) {
    auto* slice = col->add_chunk_slices();
    slice->set_chunk_key(chunk.chunk_key());
    slice->set_offset(0);
    slice->set_length(chunk.data().tensors(0).shape().dim(0));
    slice->set_index(0);
  }
  return item;
}

std::unique_ptr<Table> MakeUniformTable(const std::string& name) {
  return std::make_unique<Table>(
      name, std::make_unique<UniformSelector>(),
      std::make_unique<FifoSelector>(), 1000, 0,
      std::make_unique<RateLimiter>(1.0, 1, -DBL_MAX, DBL_MAX));
}

std::string MakeRoot() {
  // ponytail: 临时目录用 mkdtemp 生成唯一名,避免跨测试冲突。
  auto base = std::filesystem::temp_directory_path() / "test_simple_ckptXXXXXX";
  std::string tmpl = base.string();
  char* dir = mkdtemp(tmpl.data());
  REVERB_CHECK(dir != nullptr) << "mkdtemp failed for " << tmpl;
  return dir;
}

TEST(SimpleCheckpointerTest, CreatesDirectoryInRoot) {
  std::string root = MakeRoot();
  SimpleCheckpointer checkpointer(root);
  std::string path;
  REVERB_ASSERT_OK(checkpointer.Save(std::vector<Table*>{}, 1, &path));
  ASSERT_TRUE(std::filesystem::exists(path));
  // path 应位于 root 之下。
  EXPECT_NE(path.find(root), std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(path) / "DONE"));
}

TEST(SimpleCheckpointerTest, SaveAndLoad) {
  ChunkStore chunk_store;

  auto table = MakeUniformTable("uniform");

  std::vector<ChunkStore::Key> chunk_keys;
  for (int i = 0; i < 2; i++) {
    chunk_keys.push_back(1000 + i);
    auto chunk = chunk_store.Insert(MakeChunkData(chunk_keys.back()));
    REVERB_EXPECT_OK(table->InsertOrAssign(
        {MakePrioritizedItem(table->name(), i, static_cast<double>(i),
                             {chunk->data()}),
         {chunk}}));
  }
  ASSERT_EQ(table->size(), 2);

  SimpleCheckpointer checkpointer(MakeRoot());
  std::string path;
  REVERB_ASSERT_OK(checkpointer.Save({table.get()}, 1, &path));

  ChunkStore loaded_chunk_store;
  std::vector<std::shared_ptr<Table>> loaded_tables;
  loaded_tables.push_back(MakeUniformTable("uniform"));

  REVERB_ASSERT_OK(checkpointer.Load(path, &loaded_chunk_store, &loaded_tables));

  // item 数量一致。
  EXPECT_EQ(loaded_tables[0]->size(), 2);

  // chunk 都已恢复到 loaded_chunk_store。
  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
  REVERB_EXPECT_OK(loaded_chunk_store.Get(chunk_keys, &chunks));

  // 每个 item 的 key/priority/table 与原表一致。`Table::Copy()` 迭代
  // `flat_hash_map`,其顺序依赖 key 的 hash 与内部布局,不保证原表与
  // 加载表一致,故按 key 建索引后逐个比较。
  auto original_items = table->Copy();
  auto loaded_items = loaded_tables[0]->Copy();
  ASSERT_EQ(original_items.size(), loaded_items.size());
  internal::flat_hash_map<uint64_t, const Table::Item*> loaded_by_key;
  for (const auto& item : loaded_items) {
    loaded_by_key[item.key()] = &item;
  }
  for (const auto& orig : original_items) {
    auto it = loaded_by_key.find(orig.key());
    ASSERT_NE(it, loaded_by_key.end());
    const Table::Item* loaded = it->second;
    EXPECT_EQ(orig.key(), loaded->key());
    EXPECT_DOUBLE_EQ(orig.priority(), loaded->priority());
    EXPECT_EQ(orig.table(), loaded->table());
    EXPECT_EQ(orig.flat_trajectory().SerializeAsString(),
              loaded->flat_trajectory().SerializeAsString());
  }
}

TEST(SimpleCheckpointerTest, LoadLatest) {
  ChunkStore chunk_store;
  auto table = MakeUniformTable("table0");
  auto chunk = chunk_store.Insert(MakeChunkData(42));
  REVERB_EXPECT_OK(table->InsertOrAssign(
      {MakePrioritizedItem(table->name(), 0, 1.0, {chunk->data()}), {chunk}}));

  std::string root = MakeRoot();
  SimpleCheckpointer checkpointer(root);
  std::string path;
  REVERB_ASSERT_OK(checkpointer.Save({table.get()}, 1, &path));

  std::vector<std::shared_ptr<Table>> loaded_tables;
  loaded_tables.push_back(MakeUniformTable("table0"));
  REVERB_ASSERT_OK(checkpointer.LoadLatest(&loaded_tables));
  EXPECT_EQ(loaded_tables[0]->size(), 1);
}

TEST(SimpleCheckpointerTest, SaveDeletesOldData) {
  ChunkStore chunk_store;
  auto table = MakeUniformTable("table0");
  auto chunk = chunk_store.Insert(MakeChunkData(42));
  REVERB_EXPECT_OK(table->InsertOrAssign(
      {MakePrioritizedItem(table->name(), 0, 1.0, {chunk->data()}), {chunk}}));

  std::string root = MakeRoot();
  SimpleCheckpointer checkpointer(root);

  for (int i = 0; i < 5; i++) {
    std::string path;
    REVERB_ASSERT_OK(checkpointer.Save({table.get()}, 2, &path));
  }

  int subdirs = 0;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (entry.is_directory()) subdirs++;
  }
  EXPECT_EQ(subdirs, 2);
}

TEST(SimpleCheckpointerTest, KeepLatestZeroReturnsError) {
  auto table = MakeUniformTable("table0");
  SimpleCheckpointer checkpointer(MakeRoot());
  std::string path;
  EXPECT_EQ(checkpointer.Save({table.get()}, 0, &path).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(SimpleCheckpointerTest, LoadLatestInEmptyDir) {
  SimpleCheckpointer checkpointer(MakeRoot());
  std::vector<std::shared_ptr<Table>> tables;
  EXPECT_EQ(checkpointer.LoadLatest(&tables).code(),
            absl::StatusCode::kNotFound);
}

TEST(SimpleCheckpointerTest, LoadMissingFallbackCheckpoint) {
  SimpleCheckpointer checkpointer(MakeRoot(), MakeRoot());
  std::vector<std::shared_ptr<Table>> tables;
  EXPECT_EQ(checkpointer.LoadFallbackCheckpoint(&tables).code(),
            absl::StatusCode::kNotFound);
}

TEST(SimpleCheckpointerTest, LoadFallbackCheckpoint) {
  ChunkStore chunk_store;
  auto table = MakeUniformTable("uniform");
  auto chunk = chunk_store.Insert(MakeChunkData(7));
  REVERB_EXPECT_OK(table->InsertOrAssign(
      {MakePrioritizedItem(table->name(), 0, 1.0, {chunk->data()}), {chunk}}));

  SimpleCheckpointer first_checkpointer(MakeRoot());
  std::string path;
  REVERB_ASSERT_OK(first_checkpointer.Save({table.get()}, 1, &path));

  SimpleCheckpointer second_checkpointer(MakeRoot(), path);
  std::vector<std::shared_ptr<Table>> loaded_tables;
  loaded_tables.push_back(MakeUniformTable("uniform"));
  REVERB_ASSERT_OK(second_checkpointer.LoadFallbackCheckpoint(&loaded_tables));
  EXPECT_EQ(loaded_tables[0]->size(), 1);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
