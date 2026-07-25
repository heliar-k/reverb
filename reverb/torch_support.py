# Copyright 2019 DeepMind Technologies Limited.
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

"""Optional torch.Tensor support for Reverb (see docs/torch-tensor-spec.md).

torch is an optional dependency: this module imports lazily and never
requires torch at import time. Conversion happens once per leaf at the
writer/sample boundary; everything below stays opaque bytes.
"""

import numpy as np
import tree

_TORCH = ...
_TORCH_UNAVAILABLE_MSG = (
    "torch is required for torch.Tensor support; "
    "install it with `pip install dm-reverb-numpy[torch]`"
)


def _torch():
    """Returns the torch module, or None if not installed. Cached."""
    global _TORCH
    if _TORCH is ...:
        try:
            import torch

            _TORCH = torch
        except ImportError:
            _TORCH = None
    return _TORCH


def is_available() -> bool:
    return _torch() is not None


def _require_torch():
    """Returns the torch module; raises ImportError with install hint if absent."""
    torch = _torch()
    if torch is None:
        raise ImportError(_TORCH_UNAVAILABLE_MSG)
    return torch


def is_tensor(x) -> bool:
    torch = _torch()
    return torch is not None and isinstance(x, torch.Tensor)


def to_numpy_leaf(x):
    """Converts a torch.Tensor leaf to a numpy array; passes others through.

    CPU tensors convert zero-copy (`.numpy()` shares memory). CUDA tensors
    incur one unavoidable D2H copy (device pointers are process-local).
    Non-contiguous tensors keep their strides; the C++ boundary applies
    `ascontiguousarray` exactly as it does for numpy today.

    Raises:
        ValueError: for torch dtypes with no numpy counterpart (e.g. bfloat16,
            quantized, sparse tensors).
        ImportError: if x is a torch.Tensor but torch is unavailable (only
            reachable via monkey-patching; isinstance guards normally prevent).
    """
    if not is_tensor(x):
        return x
    t = x.detach()
    if t.is_cuda:
        t = t.cpu()
    try:
        return t.numpy()
    except (TypeError, RuntimeError) as e:
        raise ValueError(
            f"torch dtype {x.dtype} has no numpy counterpart and cannot be "
            f"written to reverb; cast it to a supported dtype first"
        ) from e


def to_numpy_tree(structure):
    """Maps `to_numpy_leaf` over a nested structure (no-op without torch)."""
    if not is_available():
        return structure
    return tree.map_structure(to_numpy_leaf, structure)


def from_numpy_leaf(x):
    """Converts a numpy leaf to a zero-copy torch.Tensor when possible.

    Leaves whose dtype torch cannot represent (e.g. strings, uint16/32/64)
    are returned unchanged (fallback stays numpy).

    Raises:
        ImportError: if x is an ndarray but torch is not installed.
    """
    if not isinstance(x, np.ndarray):
        return x
    try:
        return _require_torch().from_numpy(x)
    except TypeError:
        return x
