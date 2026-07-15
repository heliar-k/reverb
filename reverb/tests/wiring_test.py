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

"""Pilot test: verify tests/ directory wires into the reverb package via bazel.

If this passes, the import path + deps for all migrated tests are correct.
"""

from absl.testing import absltest

import reverb
from reverb import errors, replay_sample


class TestsWiringTest(absltest.TestCase):
    def test_package_imports(self):
        # Sanity: the public package surface is reachable from tests/.
        self.assertTrue(hasattr(reverb, "Client"))
        self.assertTrue(hasattr(reverb, "LocalClient"))
        self.assertTrue(hasattr(reverb, "Server"))
        self.assertTrue(hasattr(reverb, "Table"))
        self.assertTrue(hasattr(reverb, "TrajectoryWriter"))
        self.assertTrue(hasattr(reverb, "TrajectoryColumn"))
        self.assertTrue(hasattr(reverb, "selectors"))
        self.assertTrue(hasattr(reverb, "rate_limiters"))
        self.assertTrue(hasattr(reverb, "InProcessClient"))

    def test_errors_hierarchy(self):
        self.assertTrue(issubclass(errors.DeadlineExceededError, errors.ReverbError))
        self.assertTrue(issubclass(errors.ReverbError, Exception))

    def test_replay_sample_zeros(self):
        info = replay_sample.SampleInfo.zeros()
        self.assertEqual(info.key, 0)
        self.assertEqual(info.probability, 0.0)
        self.assertEqual(info.table_size, 0)
        self.assertEqual(info.priority, 0.0)
        self.assertEqual(info.times_sampled, 0)


if __name__ == "__main__":
    absltest.main()
