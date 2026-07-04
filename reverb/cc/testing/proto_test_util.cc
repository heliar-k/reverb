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

#include "reverb/cc/testing/proto_test_util.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/tensor_compression.h"

namespace deepmind {
namespace reverb {
namespace testing {

namespace {

// Builds a DT_INT32 TensorBuffer of the given shape filled with 1, mirroring
// the historical `tensorflow::Tensor(DT_INT32, shape).setConstant(1)` helper
// without pulling in TF.
TensorBuffer MakeInt32Filled(const std::vector<int64_t>& shape) {
  int64_t num = 1;
  for (int64_t d : shape) num *= d;
  std::string bytes(sizeof(int32_t) * num, '\0');
  int32_t one = 1;
  for (int64_t i = 0; i < num; ++i) {
    std::memcpy(&bytes[i * sizeof(int32_t)], &one, sizeof(int32_t));
  }
  return TensorBuffer(TensorSpec{DataType::Int32, shape}, std::move(bytes));
}

}  // namespace

ChunkData MakeChunkData(uint64_t key) {
  return MakeChunkData(key, MakeSequenceRange(key * 100, 0, 1), 1);
}

ChunkData MakeChunkData(uint64_t key, SequenceRange range) {
  return MakeChunkData(key, range, 1);
}

ChunkData MakeChunkData(uint64_t key, SequenceRange range, int num_tensors) {
  ChunkData chunk;
  chunk.set_chunk_key(key);
  TensorBuffer t = MakeInt32Filled(
      {range.end() - range.start() + 1, 10});
  for (int i = 0; i < num_tensors; i++) {
    CHECK_OK(CompressTensorAsProto(t, chunk.mutable_data()->add_tensors()));
  }
  *chunk.mutable_sequence_range() = std::move(range);

  return chunk;
}

SequenceRange MakeSequenceRange(uint64_t episode_id, int32_t start,
                                int32_t end) {
  REVERB_CHECK_LE(start, end);
  SequenceRange sequence;
  sequence.set_episode_id(episode_id);
  sequence.set_start(start);
  sequence.set_end(end);
  return sequence;
}

KeyWithPriority MakeKeyWithPriority(uint64_t key, double priority) {
  KeyWithPriority update;
  update.set_key(key);
  update.set_priority(priority);
  return update;
}

PrioritizedItem MakePrioritizedItem(uint64_t key, double priority,
                                    const std::vector<ChunkData>& chunks) {
  QCHECK(!chunks.empty());

  PrioritizedItem item;
  item.set_key(key);
  item.set_priority(priority);

  for (int i = 0; i < chunks.front().data().tensors_size(); i++) {
    auto* col = item.mutable_flat_trajectory()->add_columns();
    for (const auto& chunk : chunks) {
      auto* slice = col->add_chunk_slices();
      slice->set_chunk_key(chunk.chunk_key());
      slice->set_offset(0);
      slice->set_length(chunk.data().tensors(0).shape().dim(0));
      slice->set_index(i);
    }
  }

  return item;
}

PrioritizedItem MakePrioritizedItem(const std::string& table, uint64_t key,
                                    double priority,
                                    const std::vector<ChunkData>& chunks) {
  auto item = MakePrioritizedItem(key, priority, chunks);
  item.set_table(table);
  return item;
}

}  // namespace testing
}  // namespace reverb
}  // namespace deepmind
