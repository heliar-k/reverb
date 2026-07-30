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

#ifndef REVERB_CC_SUPPORT_SIGNATURE_H_
#define REVERB_CC_SUPPORT_SIGNATURE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/default/hash_map.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace internal {

// Description of a single tensor column. `shape` uses -1 for an unknown
// (wildcard) dimension, mirroring the semantics of TF's PartialTensorShape
// without pulling in TensorFlow.
struct TensorSpec {
  std::string name;
  DataType dtype;
  std::vector<int64_t> shape;

  // Two specs are compatible iff they have the same rank and, per dimension,
  // either side is -1 (wildcard) or the sizes are equal. `dtype` must match.
  bool IsCompatibleWith(const TensorSpec& other) const;
  std::string DebugString() const;
};

typedef absl::optional<std::vector<TensorSpec>> DtypesAndShapes;

absl::Status FlatSignatureFromTableInfo(
    const TableInfo& info, DtypesAndShapes* dtypes_and_shapes);

absl::Status FlatSignatureFromSignatureProto(
    const ::reverb::tensor::SignatureProto& value,
    DtypesAndShapes* dtypes_and_shapes);

absl::Status AddBatchDim(::reverb::tensor::SignatureProto* value,
                         int batch_size);

// Infers a SignatureProto (a ListValue of TensorSpecs) from the tensors stored
// in `chunk_data`. The leading (batch) dimension is dropped from each spec.
::reverb::tensor::SignatureProto SignatureProtoFromChunkData(
    const ChunkData& chunk_data);

// Create a SignatureProto of the trajectory referenced by `item`. Non
// squeezed columns are assigned a batch dimension of -1.
::reverb::tensor::SignatureProto SignatureProtoFromItem(const TableItem& item);

// Map from table name to optional vector of flattened (dtype, shape) pairs.
typedef internal::flat_hash_map<std::string, internal::DtypesAndShapes>
    FlatSignatureMap;

std::string DtypesShapesString(
    const std::vector<internal::TensorSpec>& dtypes_and_shapes);

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_SIGNATURE_H_
