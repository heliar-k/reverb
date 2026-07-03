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

#ifndef REVERB_CC_PLATFORM_DEFAULT_SIMPLE_CHECKPOINTER_H_
#define REVERB_CC_PLATFORM_DEFAULT_SIMPLE_CHECKPOINTER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {

// SimpleCheckpointer 是 TFRecordCheckpointer 的去 TF 替代实现。复刻了原实现
// 的目录结构与序列化契约,仅把 TFRecord(RecordWriter/RecordReader + zlib)换成
// Task 3 的 length-delimited protobuf IO。
//
// Checkpoint 目录布局(与 TFRecordCheckpointer 一致,以便互操作 / 等价语义):
//
//   <root_path>/
//     <timestamp of the checkpoint>/   // absl::FormatTime(absl::Now())
//       tables.ckpt                    // 多条 PriorityTableCheckpoint
//       items.ckpt                     // 多条 PrioritizedItem
//       chunks.ckpt                    // 多条 ChunkData
//       DONE                           // 空文件,写完所有数据后创建
//
// DONE 不存在 => checkpoint 正在写或被中断,视为损坏,Load 拒绝。
// 最新的 checkpoint 由 root_path 下子目录名(时间戳字符串)排序得到。
//
// 可选 `fallback_checkpoint_path`:root_path 下无 checkpoint 时,从该路径加载
// (用于用其他实验的 checkpoint 初始化服务)。
class SimpleCheckpointer : public Checkpointer {
 public:
  explicit SimpleCheckpointer(
      absl::string_view root_path,
      absl::optional<std::string> fallback_checkpoint_path = absl::nullopt);

  // Save 一个新 checkpoint 到 root_path_ 子目录。成功后 `path` 返回该子目录的
  // 绝对路径。若 root_path_ 不存在会递归创建。保存后只保留 keep_latest 个最近
  // checkpoint,其余删除。
  absl::Status Save(std::vector<Table*> tables, int keep_latest,
                    std::string* path) override;

  // 从 `path` 加载 checkpoint。`tables` 中的 Table 必须已存在(按 name 匹配),
  // 加载时用旧 table 的 extensions,替换其内部状态。
  absl::Status Load(absl::string_view path, ChunkStore* chunk_store,
                    std::vector<std::shared_ptr<Table>>* tables) override;

  // 找 root_path_ 下最新(目录名最大)且有 DONE 的 checkpoint,调 Load。
  absl::Status LoadLatest(std::vector<std::shared_ptr<Table>>* tables) override;

  // 加载 fallback_checkpoint_path_(若设置且存在 DONE)。
  absl::Status LoadFallbackCheckpoint(
      std::vector<std::shared_ptr<Table>>* tables) override;

  std::string DebugString() const override;

  SimpleCheckpointer(const SimpleCheckpointer&) = delete;
  SimpleCheckpointer& operator=(const SimpleCheckpointer&) = delete;

 private:
  const std::string root_path_;
  absl::optional<std::string> fallback_checkpoint_path_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_PLATFORM_DEFAULT_SIMPLE_CHECKPOINTER_H_
