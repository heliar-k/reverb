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

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/table.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind {
namespace reverb {

InProcessClient::InProcessClient(std::vector<std::shared_ptr<Table>> tables,
                                 std::shared_ptr<Checkpointer> checkpointer)
    : checkpointer_(std::move(checkpointer)) {
  for (auto& table : tables) {
    REVERB_CHECK(table != nullptr) << "InProcessClient: null table provided.";
    tables_[table->name()] = std::move(table);
  }
}

std::shared_ptr<Table> InProcessClient::GetTable(absl::string_view name) const {
  auto it = tables_.find(name);
  return it == tables_.end() ? nullptr : it->second;
}

absl::Status InProcessClient::NewTrajectoryWriter(
    const std::string& table, const TrajectoryWriter::Options& options,
    std::unique_ptr<TrajectoryWriter>* writer) {
  REVERB_RETURN_IF_ERROR(options.Validate());
  auto table_ptr = GetTable(table);
  if (table_ptr == nullptr) {
    return absl::NotFoundError(absl::StrCat(
        "InProcessClient::NewTrajectoryWriter: table '", table,
        "' not found."));
  }
  // 本地路径:TrajectoryWriter 构造时绑定单一 table_,所有插入走
  // InsertOrAssignAsync 到该 table。options.flat_signature_map 由调用方按需
  // 填充(进程内直连无服务端,默认 nullopt 即跳过 signature 校验)。
  *writer = std::make_unique<TrajectoryWriter>(std::move(table_ptr), options);
  return absl::OkStatus();
}

absl::Status InProcessClient::NewSampler(const std::string& table_name,
                                         const Sampler::Options& options,
                                         std::unique_ptr<Sampler>* sampler) {
  REVERB_RETURN_IF_ERROR(options.Validate());
  auto table = GetTable(table_name);
  if (table == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("InProcessClient::NewSampler: table '", table_name,
                     "' not found."));
  }
  *sampler = std::make_unique<Sampler>(std::move(table), options,
                                       /*dtypes_and_shapes=*/absl::nullopt);
  return absl::OkStatus();
}

absl::Status InProcessClient::MutatePriorities(
    absl::string_view table, const std::vector<KeyWithPriority>& updates,
    const std::vector<uint64_t>& deletes) {
  auto t = GetTable(table);
  if (t == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("InProcessClient::MutatePriorities: table '", table,
                     "' not found."));
  }
  return t->MutateItems(absl::MakeConstSpan(updates),
                        absl::MakeConstSpan(deletes));
}

absl::Status InProcessClient::Reset(absl::string_view table) {
  auto t = GetTable(table);
  if (t == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("InProcessClient::Reset: table '", table, "' not found."));
  }
  return t->Reset();
}

absl::Status InProcessClient::Checkpoint(std::string* path) {
  if (checkpointer_ == nullptr) {
    return absl::FailedPreconditionError(
        "InProcessClient::Checkpoint: no checkpointer provided.");
  }
  std::vector<Table*> raw_tables;
  raw_tables.reserve(tables_.size());
  for (auto& [_, table] : tables_) {
    raw_tables.push_back(table.get());
  }
  return checkpointer_->Save(std::move(raw_tables), /*keep_latest=*/1, path);
}

absl::Status InProcessClient::ServerInfo(std::vector<TableInfo>* table_info) {
  table_info->clear();
  table_info->reserve(tables_.size());
  for (auto& [_, table] : tables_) {
    table_info->push_back(table->info());
  }
  return absl::OkStatus();
}

}  // namespace reverb
}  // namespace deepmind
