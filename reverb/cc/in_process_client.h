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

#ifndef REVERB_CC_IN_PROCESS_CLIENT_H_
#define REVERB_CC_IN_PROCESS_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/table.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind {
namespace reverb {

// 进程内 Client:直接持有 `Table`,所有方法零 gRPC、零网络。
// API 对齐 `Client` 的工厂方法,供内嵌模式使用。
// `TrajectoryWriter`/`Sampler` 直接复用 Task 5/9 引入的本地路径构造函数。
class InProcessClient {
 public:
  // `tables` 为本 client 直接管理的表;key 取 `Table::name()`。
  // `checkpointer` 可为 nullptr,此时 `Checkpoint` 返回 FailedPreconditionError。
  explicit InProcessClient(
      std::vector<std::shared_ptr<Table>> tables,
      std::shared_ptr<Checkpointer> checkpointer = nullptr);

  // 校验 `options` 并通过本地构造函数创建 `TrajectoryWriter`,绑定到 `table`。
  // 本地 TrajectoryWriter 在构造时绑定单一 table,所有 `CreateItem` 写入该
  // table(`CreateItem` 的 table 名参数仅用于 signature 校验,本地路径不校验)。
  // 与 `Client::NewTrajectoryWriter` 的区别:进程内直连必须显式指定目标 table。
  absl::Status NewTrajectoryWriter(const std::string& table,
                                   const TrajectoryWriter::Options& options,
                                   std::unique_ptr<TrajectoryWriter>* writer);

  // 通过本地构造函数创建 `Sampler`,直接从 `table` 采样。
  // 不做 signature 校验(dtypes_and_shapes = nullopt),与
  // `Client::NewSamplerWithoutSignatureCheck` 等价。
  absl::Status NewSampler(const std::string& table_name,
                          const Sampler::Options& options,
                          std::unique_ptr<Sampler>* sampler);

  // 直接调 `Table::MutateItems`。
  absl::Status MutatePriorities(absl::string_view table,
                                const std::vector<KeyWithPriority>& updates,
                                const std::vector<uint64_t>& deletes);

  // 直接调 `Table::Reset`。
  absl::Status Reset(absl::string_view table);

  // 调 `checkpointer_->Save`(若未提供 checkpointer 则报错)。
  absl::Status Checkpoint(std::string* path);

  // 聚合所有 table 的 `info()`。
  absl::Status ServerInfo(std::vector<TableInfo>* table_info);

 private:
  // 按名查表,找不到返回 nullptr。
  std::shared_ptr<Table> GetTable(absl::string_view name) const;

  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables_;
  std::shared_ptr<Checkpointer> checkpointer_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_IN_PROCESS_CLIENT_H_
