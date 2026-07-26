# Copyright 2026 DeepMind Technologies Limited.
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

"""Write-path zero-copy (REVERB_ZERO_COPY_APPEND=1) end-to-end contract.

This target runs with the env var set (see tests/BUILD): appended numeric
arrays are viewed, not copied, and the source is marked read-only so an
in-place rewrite fails loudly instead of silently corrupting replay data.
"""

import numpy as np
from absl.testing import absltest

import reverb


def _make_client():
    server = reverb.Server(
        tables=[
            reverb.Table(
                name="t",
                sampler=reverb.selectors.Fifo(),
                remover=reverb.selectors.Fifo(),
                max_size=10,
                max_times_sampled=1,
                rate_limiter=reverb.rate_limiters.MinSize(1),
            )
        ],
        in_process=True,
    )
    return server.in_process_client


class ZeroCopyWriteTest(absltest.TestCase):
    def test_source_marked_readonly_after_append(self):
        client = _make_client()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": arr})
        self.assertFalse(arr.flags.writeable)

    def test_inplace_mutation_after_append_raises(self):
        client = _make_client()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": arr})
        with self.assertRaises(ValueError):
            arr[0] = 9.0

    def test_roundtrip_values(self):
        client = _make_client()
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": np.array([1.0, 2.0], dtype=np.float32)})
            w.create_item(
                table="t", priority=1.0, trajectory={"obs": w.history["obs"][:]}
            )
            w.flush()
        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        np.testing.assert_allclose(samples[0].data[0], [[1.0, 2.0]])


if __name__ == "__main__":
    absltest.main()
