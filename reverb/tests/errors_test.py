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

"""Tests for reverb.errors."""

from absl.testing import absltest

from reverb import errors


class ErrorsTest(absltest.TestCase):

  def test_reverb_error_is_exception(self):
    self.assertTrue(issubclass(errors.ReverbError, Exception))

  def test_deadline_exceeded_is_reverb_error(self):
    self.assertTrue(issubclass(errors.DeadlineExceededError, errors.ReverbError))

  def test_reverb_error_raisable(self):
    with self.assertRaises(errors.ReverbError):
      raise errors.ReverbError('boom')

  def test_deadline_exceeded_raisable_as_reverb_error(self):
    # DeadlineExceededError must be catchable as the base ReverbError.
    with self.assertRaises(errors.ReverbError):
      raise errors.DeadlineExceededError('timeout')

  def test_deadline_exceeded_raisable_as_itself(self):
    with self.assertRaises(errors.DeadlineExceededError):
      raise errors.DeadlineExceededError('timeout')

  def test_message_preserved(self):
    err = errors.DeadlineExceededError('timed out after 5s')
    self.assertIn('timed out', str(err))


if __name__ == '__main__':
  absltest.main()
