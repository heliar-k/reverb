# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-Python codec for `reverb.tensor.SignatureProto`.

Replaces TF's `nested_structure_coder` for table signatures. Encodes a nested
Python structure (dict / list / tuple / namedtuple with `TensorSpec` leaves)
to/from the same `SignatureProto` the C++ layer uses (see
`third_party/reverb_tensor/reverb_tensor.proto`).

dtype mapping mirrors `reverb/cc/support/tensor_proxy.cc` `kMappings`
(numpy dtype <-> `reverb.tensor.DataType`). Shape dimensions may be `None`
(unknown), encoded as `-1` on the wire (matching the C++ batch-dim convention).
"""

import collections
from typing import Any, Optional, Tuple

import numpy as np

from third_party.reverb_tensor import reverb_tensor_pb2

# numpy dtype -> reverb.tensor.DataType enum value name.
# ponytail: covers all DataType variants in reverb_tensor.proto; new dtypes add
# one row here AND in _DTYPE_NAME_TO_NP. No partial coverage.
_NP_TO_DT_NAME = {
    np.dtype(np.float32): "DT_FLOAT32",
    np.dtype(np.float64): "DT_FLOAT64",
    np.dtype(np.int8): "DT_INT8",
    np.dtype(np.int16): "DT_INT16",
    np.dtype(np.int32): "DT_INT32",
    np.dtype(np.int64): "DT_INT64",
    np.dtype(np.uint8): "DT_UINT8",
    np.dtype(np.uint16): "DT_UINT16",
    np.dtype(np.uint32): "DT_UINT32",
    np.dtype(np.uint64): "DT_UINT64",
    np.dtype(np.bool_): "DT_BOOL",
    np.dtype(np.complex64): "DT_COMPLEX64",
    np.dtype(np.complex128): "DT_COMPLEX128",
    np.dtype(np.object_): "DT_STRING",
    np.dtype(np.bytes_): "DT_STRING",
    np.dtype(np.str_): "DT_STRING",
}

# reverb.tensor.DataType enum value name -> numpy dtype.
_DT_NAME_TO_NP = {
    "DT_FLOAT32": np.float32,
    "DT_FLOAT64": np.float64,
    "DT_INT8": np.int8,
    "DT_INT16": np.int16,
    "DT_INT32": np.int32,
    "DT_INT64": np.int64,
    "DT_UINT8": np.uint8,
    "DT_UINT16": np.uint16,
    "DT_UINT32": np.uint32,
    "DT_UINT64": np.uint64,
    "DT_BOOL": np.bool_,
    "DT_COMPLEX64": np.complex64,
    "DT_COMPLEX128": np.complex128,
    "DT_STRING": np.object_,
    "DT_INVALID": np.dtype(np.object_),  # ponytail: shouldn't happen; fallback
}


def _normalize_dtype(dtype: Any) -> np.dtype:
    """Accept numpy dtype, numpy scalar type, or string; return np.dtype."""
    if isinstance(dtype, str):
        return np.dtype(dtype)
    return np.dtype(dtype)


def _np_dtype_to_proto_dtype(np_dtype: np.dtype) -> int:
    name = _NP_TO_DT_NAME.get(np_dtype)
    if name is None:
        raise ValueError(f"Unsupported numpy dtype for signature: {np_dtype!r}")
    return reverb_tensor_pb2.DataType.Value(name)


def _proto_dtype_to_np_dtype(proto_dtype: int) -> np.dtype:
    name = reverb_tensor_pb2.DataType.Name(proto_dtype)
    np_type = _DT_NAME_TO_NP.get(name)
    if np_type is None:
        raise ValueError(f"Unsupported proto DataType: {proto_dtype!r}")
    return np.dtype(np_type)


class TensorSpec:
    """Lightweight tensor spec: name, dtype (numpy), shape (tuple)."""

    __slots__ = ("name", "dtype", "shape")

    def __init__(
        self,
        shape: Tuple[Optional[int], ...],
        dtype: Any,
        name: Optional[str] = None,
    ):
        self.shape = tuple(shape)
        self.dtype = _normalize_dtype(dtype)
        self.name = name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorSpec):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and self.name == other.name
        )

    def __hash__(self) -> int:
        return hash((self.shape, self.dtype, self.name))

    def __repr__(self) -> str:
        return (
            f"TensorSpec(shape={self.shape!r}, dtype={self.dtype!r}, "
            f"name={self.name!r})"
        )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _is_namedtuple(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and hasattr(value, "_fields")
        and isinstance(getattr(value, "_fields", None), tuple)
        and all(isinstance(f, str) for f in value._fields)
    )


def _encode_leaf(spec: TensorSpec) -> reverb_tensor_pb2.SignatureProto:
    if not isinstance(spec, TensorSpec):
        raise ValueError(f"Unsupported signature leaf (expected TensorSpec): {spec!r}")
    proto = reverb_tensor_pb2.SignatureProto()
    ts = proto.tensor_spec
    ts.name = spec.name or ""
    ts.dtype = _np_dtype_to_proto_dtype(spec.dtype)
    for dim in spec.shape:
        # None (unknown dim) encoded as -1, matching C++ batch-dim convention.
        ts.shape.dim.append(-1 if dim is None else int(dim))
    return proto


def encode_signature(structure: Any) -> bytes:
    """Encode a nested structure into serialized `SignatureProto` bytes."""
    return _encode(structure).SerializeToString()


def _encode(structure: Any) -> reverb_tensor_pb2.SignatureProto:
    if isinstance(structure, TensorSpec):
        return _encode_leaf(structure)
    if isinstance(structure, dict):
        proto = reverb_tensor_pb2.SignatureProto()
        for key, value in structure.items():
            proto.dict_value.values[key].CopyFrom(_encode(value))
        return proto
    if _is_namedtuple(structure):
        proto = reverb_tensor_pb2.SignatureProto()
        nt = proto.named_tuple_value
        nt.name = type(structure).__name__
        nt.keys.extend(structure._fields)
        for value in structure:
            nt.values.add().CopyFrom(_encode(value))
        return proto
    if isinstance(structure, list):
        proto = reverb_tensor_pb2.SignatureProto()
        for value in structure:
            proto.list_value.values.add().CopyFrom(_encode(value))
        return proto
    if isinstance(structure, tuple):
        proto = reverb_tensor_pb2.SignatureProto()
        for value in structure:
            proto.tuple_value.values.add().CopyFrom(_encode(value))
        return proto
    raise ValueError(f"Unsupported signature node: {structure!r}")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode_signature(data: bytes) -> Any:
    """Decode serialized `SignatureProto` bytes back into a nested structure."""
    proto = reverb_tensor_pb2.SignatureProto.FromString(data)
    return _decode(proto)


def _decode(proto: reverb_tensor_pb2.SignatureProto) -> Any:
    kind = proto.WhichOneof("kind")
    if kind == "tensor_spec":
        ts = proto.tensor_spec
        shape = tuple(None if d == -1 else d for d in ts.shape.dim)
        return TensorSpec(
            shape=shape,
            dtype=_proto_dtype_to_np_dtype(ts.dtype),
            name=ts.name or None,
        )
    if kind == "bounded_tensor_spec":
        # ponytail: BoundedTensorSpec leaf not produced by this codec's encoder;
        # decode as a plain TensorSpec (bounds dropped). Add when callers need them.
        ts = proto.bounded_tensor_spec
        shape = tuple(None if d == -1 else d for d in ts.shape.dim)
        return TensorSpec(
            shape=shape,
            dtype=_proto_dtype_to_np_dtype(ts.dtype),
            name=ts.name or None,
        )
    if kind == "dict_value":
        return {k: _decode(v) for k, v in proto.dict_value.values.items()}
    if kind == "list_value":
        return [_decode(v) for v in proto.list_value.values]
    if kind == "tuple_value":
        return tuple(_decode(v) for v in proto.tuple_value.values)
    if kind == "named_tuple_value":
        nt = proto.named_tuple_value
        # Reconstruct as collections.namedtuple to preserve _fields semantics.
        fields = list(nt.keys) if nt.keys else [f"_{i}" for i in range(len(nt.values))]
        cls = collections.namedtuple(
            nt.name or "SignatureNamedTuple", fields, rename=True
        )
        return cls(*[_decode(v) for v in nt.values])
    # No kind set: treat as a bare-leaf placeholder (None). This matches the
    # legacy encode_structure path in structured_writer.py whose leaves are None.
    return None


if __name__ == "__main__":
    # Round-trip self-check.
    spec = {
        "a": TensorSpec([3, 3], np.float32, "a"),
        "b": {
            "c": TensorSpec([], np.int32, "b/c"),
            "d": [
                TensorSpec([None, 2], np.int64, "b/d/0"),
                TensorSpec([6], np.uint8, "b/d/1"),
            ],
        },
        "e": (TensorSpec([1], np.bool_), TensorSpec([], np.float64)),
    }
    encoded = encode_signature(spec)
    decoded = decode_signature(encoded)
    assert decoded == spec, f"round-trip mismatch:\n  want={spec!r}\n  got ={decoded!r}"
    print("signature_codec round-trip OK")
