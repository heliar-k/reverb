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

"""Exhaustive tests for Reverb rate limiters."""

import sys

from absl.testing import absltest
from absl.testing import parameterized
from reverb import rate_limiters


class TestSampleToInsertRatio(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'less_than_samples_per_insert',
          'samples_per_insert': 5,
          'error_buffer': 4,
          'want': ValueError,
      },
      {
          'testcase_name': 'less_than_one',
          'samples_per_insert': 0.5,
          'error_buffer': 0.9,
          'want': ValueError,
      },
      {
          'testcase_name': 'valid',
          'samples_per_insert': 0.5,
          'error_buffer': 1.1,
          'want': None,
      },
  )
  def test_validates_single_number_error_buffer(self, samples_per_insert,
                                                error_buffer, want):
    if want:
      with self.assertRaises(want):
        rate_limiters.SampleToInsertRatio(samples_per_insert, 10, error_buffer)
    else:  # Should not raise any error.
      rate_limiters.SampleToInsertRatio(samples_per_insert, 10, error_buffer)

  @parameterized.named_parameters(
      {
          'testcase_name': 'range_too_small_due_to_sample_per_insert_ratio',
          'min_size_to_sample': 10,
          'samples_per_insert': 5,
          'error_buffer': (8, 12),
          'want': ValueError,
      },
      {
          'testcase_name': 'range_smaller_than_2',
          'min_size_to_sample': 10,
          'samples_per_insert': 0.1,
          'error_buffer': (9.5, 10.5),
          'want': ValueError,
      },
      {
          'testcase_name': 'range_below_min_size_to_sample',
          'min_size_to_sample': 10,
          'samples_per_insert': 1,
          'error_buffer': (5, 9),
          'want': None,
      },
      {
          'testcase_name': 'range_above_min_size_to_sample',
          'min_size_to_sample': 10,
          'samples_per_insert': 1,
          'error_buffer': (11, 15),
          'want': ValueError,
      },
      {
          'testcase_name': 'min_size_to_sample_smaller_than_1',
          'min_size_to_sample': 0,
          'samples_per_insert': 1,
          'error_buffer': (-100, 100),
          'want': ValueError,
      },
      {
          'testcase_name': 'valid',
          'min_size_to_sample': 10,
          'samples_per_insert': 1,
          'error_buffer': (7, 12),
          'want': None,
      },
  )
  def test_validates_explicit_range_error_buffer(self, min_size_to_sample,
                                                 samples_per_insert,
                                                 error_buffer, want):
    if want:
      with self.assertRaises(want):
        rate_limiters.SampleToInsertRatio(samples_per_insert,
                                          min_size_to_sample, error_buffer)
    else:  # Should not raise any error.
      rate_limiters.SampleToInsertRatio(samples_per_insert, min_size_to_sample,
                                        error_buffer)

  @parameterized.named_parameters(
      {
          'testcase_name': 'valid',
          'samples_per_insert': 1,
          'want': None,
      },
      {
          'testcase_name': 'zero',
          'samples_per_insert': 0,
          'want': ValueError,
      },
      {
          'testcase_name': 'negative',
          'samples_per_insert': -1,
          'want': ValueError,
      },
  )
  def test_validates_samples_per_insert(self, samples_per_insert, want):
    if want:
      with self.assertRaises(want):
        rate_limiters.SampleToInsertRatio(samples_per_insert, 10, 100)
    else:
      rate_limiters.SampleToInsertRatio(samples_per_insert, 10, 100)

  def test_single_number_error_buffer_expands_to_range(self):
    # error_buffer=number -> range centered on spi*min_size.
    rl = rate_limiters.SampleToInsertRatio(0.5, 10, 1.1)
    self.assertAlmostEqual(rl._samples_per_insert, 0.5)
    self.assertEqual(rl._min_size_to_sample, 10)
    self.assertAlmostEqual(rl._min_diff, 0.5 * 10 - 1.1)
    self.assertAlmostEqual(rl._max_diff, 0.5 * 10 + 1.1)

  def test_explicit_range_used_verbatim(self):
    rl = rate_limiters.SampleToInsertRatio(1.0, 10, (7, 12))
    self.assertEqual(rl._min_diff, 7)
    self.assertEqual(rl._max_diff, 12)

  def test_int_error_buffer_treated_as_single_number(self):
    # int is an instance of int (not float); the code checks int|float.
    rl = rate_limiters.SampleToInsertRatio(1.0, 10, 5)
    self.assertAlmostEqual(rl._min_diff, 10 * 1.0 - 5)
    self.assertAlmostEqual(rl._max_diff, 10 * 1.0 + 5)

  def test_internal_limiter_constructed(self):
    rl = rate_limiters.SampleToInsertRatio(1.0, 10, 5)
    self.assertIsNotNone(rl.internal_limiter)


class TestMinSize(parameterized.TestCase):

  @parameterized.parameters(
      (-1, True),
      (0, True),
      (1, False),
      (100, False),
  )
  def test_raises_if_min_size_lt_1(self, min_size_to_sample, want_error):
    if want_error:
      with self.assertRaises(ValueError):
        rate_limiters.MinSize(min_size_to_sample)
    else:
      rate_limiters.MinSize(min_size_to_sample)

  def test_min_size_config(self):
    rl = rate_limiters.MinSize(5)
    self.assertEqual(rl._min_size_to_sample, 5)
    self.assertAlmostEqual(rl._samples_per_insert, 1.0)
    self.assertEqual(rl._min_diff, -sys.float_info.max)
    self.assertEqual(rl._max_diff, sys.float_info.max)


class TestQueue(parameterized.TestCase):

  @parameterized.parameters(
      (1, False),
      (10, False),
      (1000, False),
  )
  def test_constructs(self, size, _):
    rate_limiters.Queue(size)

  def test_config(self):
    rl = rate_limiters.Queue(7)
    self.assertEqual(rl._min_size_to_sample, 1)
    self.assertAlmostEqual(rl._samples_per_insert, 1.0)
    self.assertEqual(rl._min_diff, 0.0)
    self.assertEqual(rl._max_diff, 7)


class TestStack(parameterized.TestCase):

  @parameterized.parameters(
      (1, False),
      (10, False),
      (1000, False),
  )
  def test_constructs(self, size, _):
    rate_limiters.Stack(size)

  def test_config(self):
    rl = rate_limiters.Stack(7)
    self.assertEqual(rl._min_size_to_sample, 1)
    self.assertAlmostEqual(rl._samples_per_insert, 1.0)
    self.assertEqual(rl._min_diff, 0.0)
    self.assertEqual(rl._max_diff, 7)


class TestRateLimiterBase(absltest.TestCase):

  def test_repr_delegates_to_internal(self):
    rl = rate_limiters.MinSize(3)
    self.assertEqual(repr(rl), repr(rl.internal_limiter))

  def test_internal_limiter_is_pybind(self):
    from reverb import pybind  # local import to avoid module-level cycle
    rl = rate_limiters.MinSize(3)
    self.assertIsInstance(rl.internal_limiter, pybind.RateLimiter)

  def test_subclasses(self):
    self.assertIsInstance(rate_limiters.MinSize(1), rate_limiters.RateLimiter)
    self.assertIsInstance(rate_limiters.Queue(1), rate_limiters.RateLimiter)
    self.assertIsInstance(rate_limiters.Stack(1), rate_limiters.RateLimiter)
    self.assertIsInstance(
        rate_limiters.SampleToInsertRatio(1.0, 10, 5),
        rate_limiters.RateLimiter)


if __name__ == '__main__':
  absltest.main()
