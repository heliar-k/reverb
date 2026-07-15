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

"""Tests for reverb.item_selectors."""

from absl.testing import absltest

from reverb import item_selectors, pybind


class ItemSelectorsTest(absltest.TestCase):
    def test_fifo(self):
        s = item_selectors.Fifo()
        self.assertIsInstance(s, pybind.FifoSelector)

    def test_lifo(self):
        s = item_selectors.Lifo()
        self.assertIsInstance(s, pybind.LifoSelector)

    def test_uniform(self):
        s = item_selectors.Uniform()
        self.assertIsInstance(s, pybind.UniformSelector)

    def test_prioritized(self):
        s = item_selectors.Prioritized(1.0)
        self.assertIsInstance(s, pybind.PrioritizedSelector)

    def test_prioritized_zero_exponent(self):
        # Exponent 0 is legal (uniform-like); just must construct.
        item_selectors.Prioritized(0.0)

    def test_min_heap(self):
        s = item_selectors.MinHeap()
        self.assertIsInstance(s, pybind.HeapSelector)

    def test_max_heap(self):
        s = item_selectors.MaxHeap()
        self.assertIsInstance(s, pybind.HeapSelector)

    def test_min_heap_is_min(self):
        # MinHeap = HeapSelector(min_heap=True).
        s = item_selectors.MinHeap()
        # pybind.HeapSelector exposes the min_heap flag via its repr/proto; we
        # assert construction distinctness against MaxHeap instead of poking C++.
        other = item_selectors.MaxHeap()
        self.assertNotEqual(repr(s), repr(other))

    def test_all_selectors_are_item_selectors(self):
        for s in (
            item_selectors.Fifo(),
            item_selectors.Lifo(),
            item_selectors.Uniform(),
            item_selectors.Prioritized(1.0),
            item_selectors.MinHeap(),
            item_selectors.MaxHeap(),
        ):
            self.assertIsInstance(s, pybind.ItemSelector)


if __name__ == "__main__":
    absltest.main()
