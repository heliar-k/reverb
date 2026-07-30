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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_macros.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/table.h"
#include "reverb/cc/trajectory_writer.h"
#include "reverb/cc/writer.h"

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
    const TrajectoryWriter::Options& options,
    std::unique_ptr<TrajectoryWriter>* writer) {
  REVERB_RETURN_IF_ERROR(options.Validate());
  if (tables_.empty()) {
    return absl::FailedPreconditionError(
        "InProcessClient::NewTrajectoryWriter: no tables managed by this "
        "client.");
  }
  // 对齐 gRPC:用各 Table 自带的 signature 填 flat_signature_map,使
  // ItemAndRefs::Validate 走与 gRPC 同一的校验路径。每张表都必须在 map 中
  // 占一项(签名 nullopt 表示该表无 signature,Validate 跳过校验);缺项会被
  // Validate 当作“未知表”拒绝。
  TrajectoryWriter::Options effective_options = options;
  internal::FlatSignatureMap signatures;
  for (const auto& [name, table] : tables_) {
    internal::DtypesAndShapes& entry = signatures[name];
    if (table->signature().has_value()) {
      REVERB_RETURN_IF_ERROR(internal::FlatSignatureFromSignatureProto(
          table->signature().value(), &entry));
    }
    // signature 为 nullopt 时,entry 保持默认 nullopt(跳过校验)。
  }
  effective_options.flat_signature_map = std::move(signatures);
  // 本地路径:writer 持 tables_ 拷贝(shared_ptr 引用计数 +1),按 item.table()
  // 分发。LoadLatest 原地改写 Table 对象不改 shared_ptr,故 writer 自动看到恢复态。
  *writer = std::make_unique<TrajectoryWriter>(tables_, effective_options);
  return absl::OkStatus();
}

absl::Status InProcessClient::NewStructuredWriter(
    std::vector<StructuredWriterConfig> configs,
    std::unique_ptr<StructuredWriter>* writer) {
  // ponytail: 主体收敛到 MakeStructuredWriter(与 Client/ShmClient 共用),
  // 仅 NewTrajectoryWriter 钩子为本客户端专有(走本地 tables_ 路径)。
  return MakeStructuredWriter(
      std::move(configs), writer,
      [this](const TrajectoryWriter::Options& options,
             std::unique_ptr<TrajectoryWriter>* trajectory_writer) {
        return NewTrajectoryWriter(options, trajectory_writer);
      });
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

absl::Status InProcessClient::NewWriter(int chunk_length, int max_timesteps,
                                        bool delta_encoded,
                                        int max_in_flight_items,
                                        std::unique_ptr<Writer>* writer) {
  if (chunk_length < 1) {
    return absl::InvalidArgumentError("chunk_length must be >= 1.");
  }
  if (max_timesteps < 1) {
    return absl::InvalidArgumentError("max_timesteps must be >= 1.");
  }
  if (max_in_flight_items < 1) {
    return absl::InvalidArgumentError("max_in_flight_items must be >= 1.");
  }
  // 本地 Writer 持 tables_ 拷贝,按 item.table() 分发。signatures 复用各 Table
  // 自带 signature(与 NewTrajectoryWriter 一致),用于 CreateItem 校验。每张表
  // 都必须在 map 中占一项(签名 nullopt 表示无 signature,跳过校验)。
  auto signatures = std::make_shared<internal::FlatSignatureMap>();
  for (const auto& [name, table] : tables_) {
    internal::DtypesAndShapes& entry = (*signatures)[name];
    if (table->signature().has_value()) {
      REVERB_RETURN_IF_ERROR(internal::FlatSignatureFromSignatureProto(
          table->signature().value(), &entry));
    }
  }
  *writer = std::make_unique<Writer>(tables_, chunk_length, max_timesteps,
                                     delta_encoded, signatures,
                                     max_in_flight_items);
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

absl::Status InProcessClient::LoadLatest() {
  if (checkpointer_ == nullptr) {
    return absl::FailedPreconditionError(
        "InProcessClient::LoadLatest: no checkpointer provided.");
  }
  std::vector<std::shared_ptr<Table>> tables;
  tables.reserve(tables_.size());
  for (auto& [_, table] : tables_) {
    tables.push_back(table);
  }
  // LoadLatest 原地改写各 Table 对象(不改 shared_ptr),故 tables_ map
  // 仍指向同一对象,无需重建。
  return checkpointer_->LoadLatest(&tables);
}

absl::Status InProcessClient::Load(absl::string_view path) {
  if (checkpointer_ == nullptr) {
    return absl::FailedPreconditionError(
        "InProcessClient::Load: no checkpointer provided.");
  }
  std::vector<std::shared_ptr<Table>> tables;
  tables.reserve(tables_.size());
  for (auto& [_, table] : tables_) {
    tables.push_back(table);
  }
  return checkpointer_->Load(path, &chunk_store_, &tables);
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
