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

#ifndef LEARNING_DEEPMIND_REPLAY_REVERB_TENSOR_COMPRESSION_H_
#define LEARNING_DEEPMIND_REPLAY_REVERB_TENSOR_COMPRESSION_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {

// Delta encodes INT8,16,32,64 and UINT8,16,32,64 tensors of dimensions >= 2.
// The first dimension is assumed to be the time step and each timestep will be
// encoded as follows: output[i] = input[i] - input[i-1]. For encoding
// `encode=true` should be passed, for decoding `encode=false`. Other dtypes
// (and tensors of dims < 2) are returned unchanged (a copy).
TensorBuffer DeltaEncode(const TensorBuffer& tensor, bool encode);

// Applies `DeltaEncode` on a vector of tensors.
std::vector<TensorBuffer> DeltaEncodeList(
    const std::vector<TensorBuffer>& tensors, bool encode);

// Compresses a TensorBuffer with snappy. The resulting `proto` must be read
// with `DecompressTensorFromProto`. Note that string tensors are not
// compressed (their string_val is stored directly).
absl::Status CompressTensorAsProto(
    const TensorBuffer& tensor, ::reverb::tensor::TensorProto* proto);

// Assumes that the TensorProto was built by calling `CompressTensorAsProto`.
absl::StatusOr<TensorBuffer> DecompressTensorFromProto(
    const ::reverb::tensor::TensorProto& proto);

}  // namespace reverb
}  // namespace deepmind

#endif  // LEARNING_DEEPMIND_REPLAY_REVERB_TENSOR_COMPRESSION_H_
