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

#include <memory>
#include <utility>

#include "reverb/cc/platform/checkpointing.h"
#include "reverb/cc/platform/default/simple_checkpointer.h"

namespace deepmind {
namespace reverb {

std::unique_ptr<Checkpointer> CreateDefaultCheckpointer(
    std::string root_dir, std::string group,
    absl::optional<std::string> fallback_checkpoint_path) {
  // ponytail: group 历史无效(TFRecordCheckpointer.Save 对非空 group 报错),
  // SimpleCheckpointer 无 group 概念,直接忽略以保持构造期行为一致。
  (void)group;
  return std::make_unique<SimpleCheckpointer>(
      std::move(root_dir), std::move(fallback_checkpoint_path));
}

}  // namespace reverb
}  // namespace deepmind
