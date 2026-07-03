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

#include "reverb/cc/tensor_compression.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/snappy.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace {

// Delta encode/decode over a flat [time, rest] layout interpreted as unsigned
// integers of width sizeof(T). The unsigned reinterpretation matches the
// original TF implementation, which bitcast to the unsigned type so that
// subtraction wraps with well-defined semantics.
template <typename U>
TensorBuffer DeltaEncodeUnsigned(const TensorBuffer& tensor, bool encode) {
  const std::vector<int64_t>& shape = tensor.shape();
  int64_t time = shape[0];
  int64_t rest = tensor.NumElements() / time;  // ponytail: shape[0] >= 1

  std::string out(tensor.bytes().size(), '\0');
  const U* src = reinterpret_cast<const U*>(tensor.bytes().data());
  U* dst = reinterpret_cast<U*>(out.data());

  // Row 0 is copied unchanged.
  for (int64_t j = 0; j < rest; j++) dst[j] = src[j];
  // For encode: dst[i] = src[i] - src[i-1].
  // For decode: dst[i] = src[i] + dst[i-1].
  for (int64_t i = 1; i < time; i++) {
    const U* prev_src = src + (i - 1) * rest;
    const U* cur_src = src + i * rest;
    U* cur_dst = dst + i * rest;
    if (encode) {
      for (int64_t j = 0; j < rest; j++) cur_dst[j] = cur_src[j] - prev_src[j];
    } else {
      U* prev_dst = dst + (i - 1) * rest;
      for (int64_t j = 0; j < rest; j++) cur_dst[j] = cur_src[j] + prev_dst[j];
    }
  }
  return TensorBuffer(TensorSpec{tensor.dtype(), shape}, std::move(out));
}

}  // namespace

TensorBuffer DeltaEncode(const TensorBuffer& tensor, bool encode) {
  if (tensor.shape().size() < 2) return tensor;

  switch (tensor.dtype()) {
    case DataType::Int8:
    case DataType::Uint8:
      return DeltaEncodeUnsigned<uint8_t>(tensor, encode);
    case DataType::Int16:
    case DataType::Uint16:
      return DeltaEncodeUnsigned<uint16_t>(tensor, encode);
    case DataType::Int32:
    case DataType::Uint32:
      return DeltaEncodeUnsigned<uint32_t>(tensor, encode);
    case DataType::Int64:
    case DataType::Uint64:
      return DeltaEncodeUnsigned<uint64_t>(tensor, encode);
    default:
      return tensor;
  }
}

std::vector<TensorBuffer> DeltaEncodeList(
    const std::vector<TensorBuffer>& tensors, bool encode) {
  std::vector<TensorBuffer> outputs;
  outputs.reserve(tensors.size());
  for (const TensorBuffer& tensor : tensors) {
    outputs.push_back(DeltaEncode(tensor, encode));
  }
  return outputs;
}

absl::Status CompressTensorAsProto(
    const TensorBuffer& tensor, ::reverb::tensor::TensorProto* proto) {
  // SerializeToProto fills dtype/shape and either tensor_content (numeric) or
  // string_val (string). For numeric dtypes we then snappy-compress
  // tensor_content in place; string tensors are left uncompressed.
  REVERB_RETURN_IF_ERROR(tensor.SerializeToProto(proto));

  if (tensor.dtype() != DataType::String) {
    std::string compressed;
    SnappyCompressFromString(proto->tensor_content(), &compressed);
    proto->set_tensor_content(std::move(compressed));
  }
  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> DecompressTensorFromProto(
    const ::reverb::tensor::TensorProto& proto) {
  // String tensors: no compression was applied, deserialize directly.
  if (proto.dtype() == ::reverb::tensor::DT_STRING) {
    return TensorBuffer::DeserializeFromProto(proto);
  }

  // Numeric: snappy-uncompress tensor_content into a mutable copy, then
  // deserialize.
  ::reverb::tensor::TensorProto inflated = proto;
  absl::StatusOr<DataType> dt = DataTypeFromProto(proto.dtype());
  if (!dt.ok()) return dt.status();

  // Determine the uncompressed byte length from the shape + dtype so snappy
  // has a correctly-sized destination buffer.
  int64_t num_elements = 1;
  for (int64_t d : proto.shape().dim()) num_elements *= d;

  // ponytail: itemsize lookup mirrors tensor_proxy's DataTypeItemsize (private).
  // If a new numeric dtype is added there, mirror it here.
  int itemsize = 0;
  switch (*dt) {
    case DataType::Float32:
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Complex64:
      itemsize = 4;
      break;
    case DataType::Float64:
    case DataType::Int64:
    case DataType::Uint64:
    case DataType::Complex128:
      itemsize = 8;
      break;
    case DataType::Int16:
    case DataType::Uint16:
      itemsize = 2;
      break;
    case DataType::Int8:
    case DataType::Uint8:
    case DataType::Bool:
      itemsize = 1;
      break;
    case DataType::String:
    case DataType::Invalid:
      return absl::InvalidArgumentError(
          absl::StrCat("DecompressTensorFromProto: unexpected dtype ",
                       DataTypeName(*dt)));
  }

  size_t uncompressed_len =
      static_cast<size_t>(num_elements) * static_cast<size_t>(itemsize);
  std::string inflated_content;
  inflated_content.resize(uncompressed_len);
  if (!SnappyUncompressToString(proto.tensor_content(), uncompressed_len,
                                inflated_content.data())) {
    return absl::InternalError(
        "DecompressTensorFromProto: snappy uncompress failed");
  }
  inflated.set_tensor_content(std::move(inflated_content));
  return TensorBuffer::DeserializeFromProto(inflated);
}

}  // namespace reverb
}  // namespace deepmind
