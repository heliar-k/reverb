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

"""Tests for reverb.torch_support (optional torch.Tensor support).

The whole file skips when torch is not installed: torch is an optional
dependency and the bazel test env does not provide it (see
docs/torch-tensor-spec.md FR3).
"""

import numpy as np
from absl.testing import absltest

from reverb import torch_support

try:
    import torch
except ImportError:
    torch = None


class TorchSupportTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        if torch is None:
            # bazel test env has no torch: module import above already exercises
            # the no-torch path (FR3), all tensor cases skip.
            self.skipTest("torch not installed")

    def test_cpu_tensor_converts_with_same_values(self):
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        arr = torch_support.to_numpy_leaf(t)
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.arange(6, dtype=np.float32).reshape(2, 3))

    def test_cpu_tensor_zero_copy(self):
        t = torch.zeros(3)
        arr = torch_support.to_numpy_leaf(t)
        t[0] = 42.0
        self.assertEqual(arr[0], 42.0)

    def test_non_tensor_passthrough(self):
        arr = np.array([1, 2, 3])
        self.assertIs(torch_support.to_numpy_leaf(arr), arr)

    def test_python_scalar_passthrough(self):
        self.assertEqual(torch_support.to_numpy_leaf(1.5), 1.5)

    def test_requires_grad_detached(self):
        t = torch.ones(2, requires_grad=True)
        arr = torch_support.to_numpy_leaf(t)
        np.testing.assert_array_equal(arr, np.ones(2))

    def test_non_contiguous_tensor(self):
        t = torch.arange(6, dtype=torch.int32).reshape(2, 3).t()
        arr = torch_support.to_numpy_leaf(t)
        np.testing.assert_array_equal(arr, np.arange(6, dtype=np.int32).reshape(2, 3).T)

    def test_zero_dim_stays_zero_dim(self):
        arr = torch_support.to_numpy_leaf(torch.tensor(3.5))
        self.assertEqual(arr.shape, ())

    def test_bfloat16_raises(self):
        t = torch.ones(2, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            torch_support.to_numpy_leaf(t)

    def test_cuda_tensor_converts_after_d2h(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        t = torch.arange(4, dtype=torch.float32, device="cuda")
        np.testing.assert_array_equal(
            torch_support.to_numpy_leaf(t), np.arange(4, dtype=np.float32)
        )


if __name__ == "__main__":
    absltest.main()
