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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/checkpointing_utils.h"
#include "reverb/cc/platform/default/hash_map.h"
#include "reverb/cc/platform/default/hash_set.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_macros.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/length_delimited_io.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"

namespace deepmind {
namespace reverb {
namespace {

constexpr char kTablesFileName[] = "tables.ckpt";
constexpr char kItemsFileName[] = "items.ckpt";
constexpr char kChunksFileName[] = "chunks.ckpt";
constexpr char kDoneFileName[] = "DONE";

// ponytail: std::filesystem 替代 tensorflow::Env 的目录/文件操作。
// 升级路径:如需跨平台抽象(如 GCS/远程)再引入抽象层,当前本地文件够用。

std::string JoinPath(const std::string& dir, absl::string_view name) {
  return dir + "/" + std::string(name);
}

absl::Status WriteDone(const std::string& dir) {
  std::ofstream ofs(JoinPath(dir, kDoneFileName));
  if (!ofs.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to create DONE file at ", dir));
  }
  return absl::OkStatus();
}

bool HasDone(const std::string& dir) {
  return std::filesystem::exists(JoinPath(dir, kDoneFileName));
}

bool HasItems(const std::string& dir) {
  return std::filesystem::exists(JoinPath(dir, kItemsFileName));
}

absl::Status GetTableIndexStatus(
    const std::vector<std::shared_ptr<Table>>& tables,
    const std::string& name, size_t* index) {
  for (size_t i = 0; i < tables.size(); i++) {
    if (tables[i]->name() == name) {
      *index = i;
      return absl::OkStatus();
    }
  }
  std::vector<std::string> table_names(tables.size());
  for (size_t i = 0; i < tables.size(); i++) {
    table_names[i] = absl::StrCat("'", tables[i]->name(), "'");
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Trying to load table '", name,
      "' but table was not found in provided list of tables. Available "
      "tables: [",
      absl::StrJoin(table_names, ", "), "]"));
}

absl::Status CheckTrajectoryFormat(const PrioritizedItem& item) {
  if (item.has_deprecated_sequence_range() && item.has_flat_trajectory()) {
    return absl::InternalError(absl::StrCat(
        "Item ", item.key(), " has both deprecated and new trajectory format: ",
        item.DebugString(), "."));
  }
  return absl::OkStatus();
}

}  // namespace

SimpleCheckpointer::SimpleCheckpointer(
    absl::string_view root_path,
    absl::optional<std::string> fallback_checkpoint_path)
    : root_path_(std::string(root_path)),
      fallback_checkpoint_path_(std::move(fallback_checkpoint_path)) {
  REVERB_LOG(REVERB_INFO) << "Initializing SimpleCheckpointer in " << root_path_
                          << (fallback_checkpoint_path_.has_value()
                                  ? absl::StrCat(
                                        " and fallback directory ",
                                        fallback_checkpoint_path_.value(), ".")
                                  : ".");
}

absl::Status SimpleCheckpointer::Save(std::vector<Table*> tables,
                                      int keep_latest, std::string* path) {
  if (keep_latest <= 0) {
    return absl::InvalidArgumentError(
        "SimpleCheckpointer must have keep_latest > 0.");
  }

  std::string dir_path =
      JoinPath(root_path_, absl::FormatTime(absl::Now()));
  std::error_code ec;
  std::filesystem::create_directories(dir_path, ec);
  if (ec) {
    return absl::InternalError(
        absl::StrCat("Failed to create checkpoint directory ", dir_path, ": ",
                     ec.message()));
  }

  internal::flat_hash_set<std::shared_ptr<ChunkStore::Chunk>> chunks;
  std::vector<PrioritizedItem> items;

  // tables.ckpt: 每个 table 一条 PriorityTableCheckpoint(length-delimited)。
  {
    std::ofstream ofs(JoinPath(dir_path, kTablesFileName), std::ios::binary);
    if (!ofs.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open ", kTablesFileName, " for writing."));
    }
    for (Table* table : tables) {
      auto checkpoint = table->Checkpoint();
      chunks.merge(checkpoint.chunks);
      items.insert(items.end(),
                   std::make_move_iterator(checkpoint.items.begin()),
                   std::make_move_iterator(checkpoint.items.end()));
      REVERB_RETURN_IF_ERROR(WriteLengthDelimited(ofs, checkpoint.checkpoint));
    }
    ofs.flush();
    if (!ofs) {
      return absl::DataLossError(
          absl::StrCat("Failed to write ", kTablesFileName, "."));
    }
  }

  // items.ckpt: 每条 PrioritizedItem。
  {
    std::ofstream ofs(JoinPath(dir_path, kItemsFileName), std::ios::binary);
    if (!ofs.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open ", kItemsFileName, " for writing."));
    }
    for (const auto& item : items) {
      REVERB_RETURN_IF_ERROR(WriteLengthDelimited(ofs, item));
    }
    ofs.flush();
    if (!ofs) {
      return absl::DataLossError(
          absl::StrCat("Failed to write ", kItemsFileName, "."));
    }
  }

  // chunks.ckpt: 每条 ChunkData(去重后的并集)。
  {
    std::ofstream ofs(JoinPath(dir_path, kChunksFileName), std::ios::binary);
    if (!ofs.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open ", kChunksFileName, " for writing."));
    }
    for (const auto& chunk : chunks) {
      REVERB_RETURN_IF_ERROR(WriteLengthDelimited(ofs, chunk->data()));
    }
    ofs.flush();
    if (!ofs) {
      return absl::DataLossError(
          absl::StrCat("Failed to write ", kChunksFileName, "."));
    }
  }

  // 数据已落盘,写 DONE 标记。
  REVERB_RETURN_IF_ERROR(WriteDone(dir_path));

  // 删除超出 keep_latest 的旧 checkpoint。目录名是时间戳字符串,字典序即时间序。
  std::vector<std::string> subdirs;
  for (const auto& entry : std::filesystem::directory_iterator(root_path_)) {
    if (entry.is_directory()) {
      subdirs.push_back(entry.path().string());
    }
  }
  std::sort(subdirs.begin(), subdirs.end());
  int history_counter = 0;
  for (auto it = subdirs.rbegin(); it != subdirs.rend(); ++it) {
    if (++history_counter > keep_latest) {
      std::error_code rm_ec;
      std::filesystem::remove_all(*it, rm_ec);
      // 删除失败不致命(旧 checkpoint 残留),仅记日志。
      if (rm_ec) {
        REVERB_LOG(REVERB_WARNING) << "Failed to delete old checkpoint " << *it
                                   << ": " << rm_ec.message();
      }
    }
  }

  *path = std::move(dir_path);
  return absl::OkStatus();
}

absl::Status SimpleCheckpointer::Load(
    absl::string_view path, ChunkStore* chunk_store,
    std::vector<std::shared_ptr<Table>>* tables) {
  const std::string dir_path(path);
  REVERB_LOG(REVERB_INFO) << "Loading checkpoint from " << dir_path;
  if (!HasDone(dir_path)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Load called with invalid checkpoint path: ", dir_path));
  }

  REVERB_LOG(REVERB_INFO)
      << "Loading and verifying metadata of the checkpointed tables.";

  // table_name -> PriorityTableCheckpoint / items。
  internal::flat_hash_map<std::string, PriorityTableCheckpoint>
      table_checkpoints;
  internal::flat_hash_map<std::string, std::vector<PrioritizedItem>>
      table_to_items;

  std::string deprecated_items;
  {
    std::ifstream ifs(JoinPath(dir_path, kTablesFileName), std::ios::binary);
    if (!ifs.is_open()) {
      return absl::NotFoundError(
          absl::StrCat("Could not open ", kTablesFileName, " in ", dir_path));
    }
    while (true) {
      PriorityTableCheckpoint checkpoint;
      absl::Status s = ReadLengthDelimited(ifs, &checkpoint);
      if (absl::IsNotFound(s)) break;  // EOF
      REVERB_RETURN_IF_ERROR(s);

      size_t table_idx;
      REVERB_RETURN_IF_ERROR(
          GetTableIndexStatus(*tables, checkpoint.table_name(), &table_idx));

      if (!checkpoint.deprecated_items().empty()) {
        auto& items = table_to_items[checkpoint.table_name()];
        items.reserve(checkpoint.deprecated_items().size());
        deprecated_items = absl::StrCat("'", checkpoint.table_name(), "'");
        for (auto& item : *checkpoint.mutable_deprecated_items()) {
          REVERB_RETURN_IF_ERROR(CheckTrajectoryFormat(item));
          items.push_back(std::move(item));
        }
        checkpoint.mutable_deprecated_items()->Clear();
      }

      REVERB_LOG(REVERB_INFO)
          << "Metadata for table '" << checkpoint.table_name()
          << "' was successfully loaded and verified.";
      std::string table_name = checkpoint.table_name();
      table_checkpoints[table_name] = std::move(checkpoint);
    }
  }

  bool non_deprecated_items = HasItems(dir_path);
  if (non_deprecated_items) {
    if (!deprecated_items.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Checkpoint loader found mix of deprecated_items field for table ",
          deprecated_items, " and items file '",
          JoinPath(dir_path, kItemsFileName), "'"));
    }

    std::ifstream ifs(JoinPath(dir_path, kItemsFileName), std::ios::binary);
    if (!ifs.is_open()) {
      return absl::NotFoundError(
          absl::StrCat("Could not open ", kItemsFileName, " in ", dir_path));
    }
    std::vector<PrioritizedItem>* items = nullptr;
    while (true) {
      PrioritizedItem item;
      absl::Status s = ReadLengthDelimited(ifs, &item);
      if (absl::IsNotFound(s)) break;  // EOF
      REVERB_RETURN_IF_ERROR(s);
      REVERB_RETURN_IF_ERROR(CheckTrajectoryFormat(item));

      if (items == nullptr || items->empty() ||
          items->at(0).table() != item.table()) {
        if (!table_checkpoints.contains(item.table())) {
          return absl::DataLossError(absl::StrCat(
              "Unable to find table '", item.table(), "' for item '",
              item.key(), "' in the set of tables loaded from metadata."));
        }
        items = &(table_to_items[item.table()]);
      }
      items->push_back(std::move(item));
    }
  }

  REVERB_LOG(REVERB_INFO)
      << "Successfully loaded and verified metadata for all ("
      << table_checkpoints.size()
      << ") tables. We'll now proceed to read the data referenced by the items "
         "in the table.";

  // 先插入 chunk 数据到 ChunkStore,确保 item 引用的 chunk 都存在。
  internal::flat_hash_map<ChunkStore::Key, std::shared_ptr<ChunkStore::Chunk>>
      chunk_by_key;
  {
    std::ifstream ifs(JoinPath(dir_path, kChunksFileName), std::ios::binary);
    if (!ifs.is_open()) {
      return absl::NotFoundError(
          absl::StrCat("Could not open ", kChunksFileName, " in ", dir_path));
    }
    while (true) {
      ChunkData chunk_data;
      absl::Status s = ReadLengthDelimited(ifs, &chunk_data);
      if (absl::IsNotFound(s)) break;  // EOF
      REVERB_RETURN_IF_ERROR(s);
      if (chunk_data.deprecated_data_size()) {
        if (!chunk_data.data().tensors().empty()) {
          return absl::InternalError(absl::StrCat(
              "Checkpoint ChunkData chunk_key=", chunk_data.chunk_key(),
              " has both data and deprecated_data."));
        }
        chunk_data.mutable_data()->mutable_tensors()->Swap(
            chunk_data.mutable_deprecated_data());
      }
      const auto key = chunk_data.chunk_key();
      chunk_by_key[key] = chunk_store->Insert(std::move(chunk_data));

      REVERB_LOG_EVERY_N(REVERB_INFO, 100)
          << "Still reading trajectory data. " << chunk_by_key.size()
          << " records have been read so far.";
    }
  }

  REVERB_LOG(REVERB_INFO)
      << "Completed reading trajectory data. We'll now start "
         "assembling the checkpointed tables.";

  std::vector<std::shared_ptr<TableExtension>> all_table_extensions;

  for (auto& checkpoint_ref : table_checkpoints) {
    auto& checkpoint = checkpoint_ref.second;
    size_t table_idx;
    REVERB_RETURN_IF_ERROR(
        GetTableIndexStatus(*tables, checkpoint.table_name(), &table_idx));
    std::shared_ptr<Table>& server_table = tables->at(table_idx);

    auto sampler = MakeSelector(checkpoint.sampler());
    auto remover = MakeSelector(checkpoint.remover());
    auto rate_limiter =
        std::make_shared<RateLimiter>(checkpoint.rate_limiter());
    auto signature = checkpoint.has_signature()
                         ? std::make_optional(checkpoint.signature())
                         : std::nullopt;

    std::vector<std::shared_ptr<TableExtension>> extensions =
        server_table->GetExtensions();
    all_table_extensions.insert(all_table_extensions.end(),
                                std::make_move_iterator(extensions.begin()),
                                std::make_move_iterator(extensions.end()));

    server_table->InitializeFromCheckpoint(
        std::move(sampler), std::move(remover), checkpoint.max_size(),
        checkpoint.max_times_sampled(), std::move(rate_limiter),
        std::move(signature), checkpoint.num_deleted_episodes(),
        checkpoint.num_unique_samples());
  }

  // 通知 extensions 更新后的 tables 列表,使其 target table 指针更新。
  // 必须在 item 加载前调用,以保证 OnInsert 正确处理。
  for (auto& extension : all_table_extensions) {
    extension->OnCheckpointLoaded(*tables);
  }

  for (auto& table : *tables) {
    for (auto& checkpoint_item : table_to_items[table->name()]) {
      if (checkpoint_item.has_deprecated_sequence_range()) {
        std::vector<std::shared_ptr<ChunkStore::Chunk>> trajectory_chunks;
        REVERB_RETURN_IF_ERROR(chunk_store->Get(
            checkpoint_item.deprecated_chunk_keys(), &trajectory_chunks));

        *checkpoint_item.mutable_flat_trajectory() =
            internal::FlatTimestepTrajectory(
                trajectory_chunks,
                checkpoint_item.deprecated_sequence_range().offset(),
                checkpoint_item.deprecated_sequence_range().length());

        checkpoint_item.clear_deprecated_sequence_range();
        checkpoint_item.clear_deprecated_chunk_keys();
      }

      std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
      REVERB_RETURN_IF_ERROR(chunk_store->Get(
          internal::GetChunkKeys(checkpoint_item.flat_trajectory()), &chunks));

      REVERB_RETURN_IF_ERROR(table->InsertCheckpointItem(
          Table::Item(std::move(checkpoint_item), std::move(chunks))));
    }

    REVERB_LOG(REVERB_INFO)
        << "Table " << table->name() << " and " << table->size()
        << " items have been successfully loaded from checkpoint at path "
        << dir_path << ".";
  }

  REVERB_LOG(REVERB_INFO) << "Successfully loaded " << table_checkpoints.size()
                          << " tables from " << dir_path;
  return absl::OkStatus();
}

absl::Status SimpleCheckpointer::LoadLatest(
    std::vector<std::shared_ptr<Table>>* tables) {
  ChunkStore chunk_store;
  REVERB_LOG(REVERB_INFO) << "Loading latest checkpoint from " << root_path_;
  std::error_code ec;
  if (!std::filesystem::exists(root_path_, ec)) {
    return absl::NotFoundError(
        absl::StrCat("No checkpoint found in ", root_path_));
  }
  std::vector<std::string> subdirs;
  for (const auto& entry : std::filesystem::directory_iterator(root_path_)) {
    if (entry.is_directory()) {
      subdirs.push_back(entry.path().string());
    }
  }
  std::sort(subdirs.begin(), subdirs.end());
  for (auto it = subdirs.rbegin(); it != subdirs.rend(); ++it) {
    if (HasDone(*it)) {
      return Load(*it, &chunk_store, tables);
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No checkpoint found in ", root_path_));
}

absl::Status SimpleCheckpointer::LoadFallbackCheckpoint(
    std::vector<std::shared_ptr<Table>>* tables) {
  ChunkStore chunk_store;
  if (!fallback_checkpoint_path_.has_value()) {
    return absl::NotFoundError("No fallback checkpoint path provided.");
  }
  if (HasDone(fallback_checkpoint_path_.value())) {
    return Load(fallback_checkpoint_path_.value(), &chunk_store, tables);
  }
  return absl::NotFoundError(absl::StrCat("No checkpoint found in ",
                                          fallback_checkpoint_path_.value()));
}

std::string SimpleCheckpointer::DebugString() const {
  return absl::StrCat("SimpleCheckpointer(root_path=", root_path_, ")");
}

}  // namespace reverb
}  // namespace deepmind
