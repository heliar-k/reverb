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

#include "reverb/cc/support/signature.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/default/logging.h"
#include "reverb/cc/platform/default/status_macros.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/tensor_proxy.h"
#include "reverb/cc/table.h"
#include "third_party/reverb_tensor/reverb_tensor.pb.h"

namespace deepmind {
namespace reverb {
namespace internal {

bool TensorSpec::IsCompatibleWith(const TensorSpec& other) const {
  if (dtype != other.dtype) return false;
  if (shape.size() != other.shape.size()) return false;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t a = shape[i];
    const int64_t b = other.shape[i];
    if (a != -1 && b != -1 && a != b) return false;
  }
  return true;
}

std::string TensorSpec::DebugString() const {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) absl::StrAppend(&s, ",");
    absl::StrAppend(&s, shape[i]);
  }
  absl::StrAppend(&s, "]");
  return absl::StrCat(DataTypeName(dtype), " ", name, " ", s);
}

absl::Status FlatSignatureFromTableInfo(
    const TableInfo& info, DtypesAndShapes* dtypes_and_shapes) {
  if (!info.has_signature()) {
    *dtypes_and_shapes = std::nullopt;
  } else {
    const auto& sig = info.signature();
    *dtypes_and_shapes = DtypesAndShapes::value_type{};
    auto status = FlatSignatureFromSignatureProto(sig, dtypes_and_shapes);
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat(status.message(), "Full signature struct: '",
                       info.signature().DebugString(), "'"));
    }
  }
  return absl::OkStatus();
}

namespace {

template <typename T>
std::string ExtendContext(absl::string_view context, T val) {
  const std::string val_string = absl::StrCat(val);
  return absl::StrCat(context,
                      ((!context.empty() && !val_string.empty()) ? "/" : ""),
                      val_string);
}

// Reads a proto TensorShapeProto into the -1-wildcard vector used by TensorSpec.
std::vector<int64_t> ShapeFromProto(
    const ::reverb::tensor::TensorShapeProto& shape) {
  std::vector<int64_t> out;
  out.reserve(shape.dim_size());
  for (int64_t d : shape.dim()) out.push_back(d);
  return out;
}

absl::Status FlatSignatureFromSignatureProto(
    const ::reverb::tensor::SignatureProto& value, absl::string_view context,
    DtypesAndShapes* dtypes_and_shapes) {
  // Engage the optional on entry: callers (e.g. InProcessClient) pass a
  // default-constructed (nullopt) optional and rely on us to populate it.
  // Without this, `(*dtypes_and_shapes)->push_back(...)` below dereferences a
  // disengaged optional — UB that manifests as a segfault inside
  // `NewTrajectoryWriter`/`NewWriter` whenever a table has a signature.
  if (!dtypes_and_shapes->has_value()) {
    *dtypes_and_shapes = std::vector<internal::TensorSpec>{};
  }
  switch (value.kind_case()) {
    case ::reverb::tensor::SignatureProto::kTensorSpec: {
      const auto& tensor_spec = value.tensor_spec();
      absl::StatusOr<DataType> dtype =
          DataTypeFromProto(tensor_spec.dtype());
      if (!dtype.ok()) return dtype.status();
      (*dtypes_and_shapes)
          ->push_back({ExtendContext(context, tensor_spec.name()), *dtype,
                       ShapeFromProto(tensor_spec.shape())});
    } break;
    case ::reverb::tensor::SignatureProto::kBoundedTensorSpec: {
      const auto& bounded_tensor_spec = value.bounded_tensor_spec();
      // This stores the dtype and shape of the boundary tensor spec. Currently,
      // the signature of such tensors only checked against these properties.
      // TODO(b/158033101): Make the signature check fully support boundaries.
      absl::StatusOr<DataType> dtype =
          DataTypeFromProto(bounded_tensor_spec.dtype());
      if (!dtype.ok()) return dtype.status();
      (*dtypes_and_shapes)
          ->push_back({ExtendContext(context, bounded_tensor_spec.name()),
                       *dtype,
                       ShapeFromProto(bounded_tensor_spec.shape())});
    } break;
    case ::reverb::tensor::SignatureProto::kListValue: {
      const auto& values = value.list_value().values();
      for (int i = 0; i < values.size(); ++i) {
        REVERB_RETURN_IF_ERROR(FlatSignatureFromSignatureProto(
            values[i], ExtendContext(context, static_cast<size_t>(i)),
            dtypes_and_shapes));
      }
    } break;
    case ::reverb::tensor::SignatureProto::kTupleValue: {
      const auto& values = value.tuple_value().values();
      for (int i = 0; i < values.size(); ++i) {
        REVERB_RETURN_IF_ERROR(FlatSignatureFromSignatureProto(
            values[i], ExtendContext(context, static_cast<size_t>(i)),
            dtypes_and_shapes));
      }
    } break;
    case ::reverb::tensor::SignatureProto::kDictValue: {
      std::vector<std::string> keys;
      keys.reserve(value.dict_value().values_size());
      for (const auto& f : value.dict_value().values()) {
        keys.push_back(f.first);
      }
      std::sort(keys.begin(), keys.end());
      for (const auto& k : keys) {
        REVERB_RETURN_IF_ERROR(FlatSignatureFromSignatureProto(
            value.dict_value().values().at(k), ExtendContext(context, k),
            dtypes_and_shapes));
      }
    } break;
    case ::reverb::tensor::SignatureProto::kNamedTupleValue: {
      const auto& nt = value.named_tuple_value();
      const auto& keys = nt.keys();
      const auto& values = nt.values();
      for (int i = 0; i < values.size(); ++i) {
        std::string key = (i < keys.size()) ? keys.Get(i) : "";
        REVERB_RETURN_IF_ERROR(FlatSignatureFromSignatureProto(
            values[i], ExtendContext(context, key), dtypes_and_shapes));
      }
    } break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Saw unsupported encoded subtree in signature: '",
                       value.DebugString(), "'"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status FlatSignatureFromSignatureProto(
    const ::reverb::tensor::SignatureProto& value,
    DtypesAndShapes* dtypes_and_shapes) {
  return FlatSignatureFromSignatureProto(value, "", dtypes_and_shapes);
}

std::string DtypesShapesString(
    const std::vector<internal::TensorSpec>& dtypes_and_shapes) {
  std::vector<std::string> strings;
  strings.reserve(dtypes_and_shapes.size());
  for (int i = 0; i < dtypes_and_shapes.size(); ++i) {
    const auto& p = dtypes_and_shapes[i];
    strings.push_back(absl::StrCat(i, ": Tensor<name: '", p.name,
                                   "', dtype: ", DataTypeName(p.dtype),
                                   ", shape: ", p.DebugString(), ">"));
  }
  return absl::StrJoin(strings, ", ");
}

::reverb::tensor::SignatureProto SignatureProtoFromChunkData(
    const ChunkData& chunk_data) {
  ::reverb::tensor::SignatureProto value;
  for (int i = 0; i < chunk_data.data().tensors_size(); i++) {
    const auto& chunk = chunk_data.data().tensors(i);
    // Drop the leading (batch) dimension, mirroring the original behaviour
    // which called PartialTensorShape(shape).RemoveDim(0).
    auto* spec =
        value.mutable_list_value()->add_values()->mutable_tensor_spec();
    spec->set_name("");
    spec->set_dtype(chunk.dtype());
    auto* out_shape = spec->mutable_shape();
    for (int d = 1; d < chunk.shape().dim_size(); ++d) {
      out_shape->add_dim(chunk.shape().dim(d));
    }
  }
  return value;
}

absl::Status AddBatchDim(::reverb::tensor::SignatureProto* value,
                         int batch_size) {
  switch (value->kind_case()) {
    case ::reverb::tensor::SignatureProto::kTensorSpec: {
      ::reverb::tensor::SignatureProto::TensorSpec spec =
          value->tensor_spec();
      spec.clear_shape();
      spec.mutable_shape()->add_dim(batch_size);
      for (int64_t d : value->tensor_spec().shape().dim()) {
        spec.mutable_shape()->add_dim(d);
      }
      *value->mutable_tensor_spec() = std::move(spec);
    } break;
    case ::reverb::tensor::SignatureProto::kBoundedTensorSpec: {
      ::reverb::tensor::SignatureProto::BoundedTensorSpec spec =
          value->bounded_tensor_spec();
      spec.clear_shape();
      spec.mutable_shape()->add_dim(batch_size);
      for (int64_t d : value->bounded_tensor_spec().shape().dim()) {
        spec.mutable_shape()->add_dim(d);
      }
      *value->mutable_bounded_tensor_spec() = std::move(spec);
    } break;
    case ::reverb::tensor::SignatureProto::kListValue: {
      for (auto& list_value : *value->mutable_list_value()->mutable_values()) {
        REVERB_RETURN_IF_ERROR(AddBatchDim(&list_value, batch_size));
      }
    } break;
    case ::reverb::tensor::SignatureProto::kTupleValue: {
      for (auto& tuple_value :
           *value->mutable_tuple_value()->mutable_values()) {
        REVERB_RETURN_IF_ERROR(AddBatchDim(&tuple_value, batch_size));
      }
    } break;
    case ::reverb::tensor::SignatureProto::kDictValue: {
      for (auto& field : *value->mutable_dict_value()->mutable_values()) {
        REVERB_RETURN_IF_ERROR(AddBatchDim(&field.second, batch_size));
      }
    } break;
    case ::reverb::tensor::SignatureProto::kNamedTupleValue: {
      for (auto& v : *value->mutable_named_tuple_value()->mutable_values()) {
        REVERB_RETURN_IF_ERROR(AddBatchDim(&v, batch_size));
      }
    } break;
    case ::reverb::tensor::SignatureProto::KIND_NOT_SET:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Saw unsupported encoded subtree in signature: '",
                       value->DebugString(), "'"));
  }
  return absl::OkStatus();
}

::reverb::tensor::SignatureProto SignatureProtoFromItem(const TableItem& item) {
  ::reverb::tensor::SignatureProto value;

  auto get_tensor = [&](const FlatTrajectory::ChunkSlice& slice)
      -> const ::reverb::tensor::TensorProto* {
    for (const auto& chunk : item.chunks()) {
      if (chunk->key() == slice.chunk_key()) {
        return &chunk->data().data().tensors(slice.index());
      }
    }
    REVERB_CHECK(false) << "Invalid item.";
    return nullptr;  // unreachable; REVERB_CHECK aborts
  };

  for (int col_idx = 0; col_idx < item.flat_trajectory().columns_size();
       col_idx++) {
    const auto& col = item.flat_trajectory().columns(col_idx);
    const auto* tensor_proto = get_tensor(col.chunk_slices(0));

    auto* spec =
        value.mutable_list_value()->add_values()->mutable_tensor_spec();
    spec->set_name("");
    spec->set_dtype(tensor_proto->dtype());
    *spec->mutable_shape() = tensor_proto->shape();

    if (col.squeeze()) {
      // Remove the leading (batch) dimension by copying the remaining dims.
      const auto& src = tensor_proto->shape().dim();
      spec->clear_shape();
      for (int d = 1; d < src.size(); ++d) spec->mutable_shape()->add_dim(src.Get(d));
    } else {
      // Replace the leading dimension with -1 (wildcard batch).
      if (spec->shape().dim_size() > 0) {
        spec->mutable_shape()->set_dim(0, -1);
      }
    }
  }

  return value;
}

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
