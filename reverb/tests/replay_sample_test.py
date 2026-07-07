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

"""Exhaustive tests for reverb.replay_sample."""

import numpy as np

from absl.testing import absltest

from reverb import pybind
from reverb import replay_sample


class SampleInfoTest(absltest.TestCase):

  def test_has_the_correct_number_of_fields(self):
    self.assertLen(replay_sample.SampleInfo._fields,
                   pybind.Sampler.NUM_INFO_TENSORS)

  def test_fields_names(self):
    self.assertEqual(
        replay_sample.SampleInfo._fields,
        ('key', 'probability', 'table_size', 'priority', 'times_sampled'))

  def test_zeros(self):
    info = replay_sample.SampleInfo.zeros()
    self.assertEqual(info.key, 0)
    self.assertEqual(info.probability, 0.0)
    self.assertEqual(info.table_size, 0)
    self.assertEqual(info.priority, 0.0)
    self.assertEqual(info.times_sampled, 0)

  def test_construct_with_scalars(self):
    info = replay_sample.SampleInfo(
        key=5, probability=0.25, table_size=10, priority=1.5, times_sampled=2)
    self.assertEqual(info.key, 5)
    self.assertAlmostEqual(info.probability, 0.25)
    self.assertEqual(info.table_size, 10)
    self.assertAlmostEqual(info.priority, 1.5)
    self.assertEqual(info.times_sampled, 2)

  def test_construct_with_arrays(self):
    # The type hints allow np.ndarray for batched sampling.
    keys = np.array([1, 2, 3], dtype=np.uint64)
    probs = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    info = replay_sample.SampleInfo(
        key=keys, probability=probs, table_size=3,
        priority=1.0, times_sampled=1)
    np.testing.assert_array_equal(info.key, keys)
    np.testing.assert_allclose(info.probability, probs)

  def test_is_namedtuple(self):
    info = replay_sample.SampleInfo.zeros()
    self.assertIsInstance(info, tuple)
    self.assertTrue(hasattr(info, '_fields'))

  def test_field_access_by_index(self):
    info = replay_sample.SampleInfo(5, 0.5, 10, 1.0, 2)
    # NamedTuple is a tuple: (key, probability, table_size, priority,
    # times_sampled).
    self.assertEqual(info[0], 5)
    self.assertEqual(info[4], 2)


class ReplaySampleTest(absltest.TestCase):

  def test_fields(self):
    self.assertEqual(replay_sample.ReplaySample._fields, ('info', 'data'))

  def test_construct(self):
    info = replay_sample.SampleInfo.zeros()
    data = [np.zeros([3])]
    sample = replay_sample.ReplaySample(info=info, data=data)
    self.assertEqual(sample.info, info)
    self.assertIs(sample.data, data)

  def test_is_namedtuple(self):
    info = replay_sample.SampleInfo.zeros()
    sample = replay_sample.ReplaySample(info=info, data=[])
    self.assertIsInstance(sample, tuple)
    self.assertTrue(hasattr(sample, '_fields'))

  def test_field_access_by_index(self):
    info = replay_sample.SampleInfo.zeros()
    data = [np.array([1, 2])]
    sample = replay_sample.ReplaySample(info=info, data=data)
    self.assertEqual(sample[0], info)
    self.assertEqual(sample[1], data)


if __name__ == '__main__':
  absltest.main()
