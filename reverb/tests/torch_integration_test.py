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

"""End-to-end torch.Tensor tests (docs/torch-tensor-spec.md FR1).

Writes torch.Tensors through every public writer API of LocalClient and
verifies the sampled numpy data matches what equivalent numpy writes would
produce. Skips entirely when torch is not installed (bazel test env).
"""

import numpy as np
from absl.testing import absltest

import reverb
from reverb import structured_writer

try:
    import torch
except ImportError:
    torch = None


def _make_server(table_name="t"):
    return reverb.Server(
        tables=[
            reverb.Table(
                name=table_name,
                sampler=reverb.selectors.Fifo(),
                remover=reverb.selectors.Fifo(),
                max_size=10,
                max_times_sampled=1,
                rate_limiter=reverb.rate_limiters.MinSize(1),
            )
        ],
        in_process=True,
    )


class TorchWritePathTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        if torch is None:
            self.skipTest("torch not installed")

    def test_trajectory_writer_accepts_torch(self):
        server = _make_server()
        client = server.in_process_client
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": torch.tensor([1.0, 2.0])})
            w.create_item("t", priority=1.0, trajectory={"obs": w.history["obs"][:]})
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]])

    def test_trajectory_writer_mixed_torch_and_numpy(self):
        server = _make_server()
        client = server.in_process_client
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append(
                {
                    "obs": torch.tensor([1.0], dtype=torch.float32),
                    "reward": np.array([0.5], dtype=np.float64),
                }
            )
            w.create_item(
                "t",
                priority=1.0,
                trajectory={
                    "obs": w.history["obs"][:],
                    "reward": w.history["reward"][:],
                },
            )
            w.flush()

        # ReplaySample.data 是打平的叶子序列(dict 按键排序): obs 在前 reward 在后。
        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0]])
        np.testing.assert_allclose(np.asarray(samples[0].data[1]), [[0.5]])

    def test_legacy_writer_accepts_torch(self):
        server = _make_server()
        client = server.in_process_client
        with client.writer(max_sequence_length=2, chunk_length=2) as w:
            w.append({"obs": torch.tensor([1.0])})
            w.append({"obs": torch.tensor([2.0])})
            w.create_item("t", num_timesteps=2, priority=1.0)
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0], [2.0]])

    def test_structured_writer_accepts_torch(self):
        server = _make_server()
        client = server.in_process_client
        pattern = structured_writer.pattern_from_transform(
            step_structure={"obs": None}, transform=lambda s: {"obs": s["obs"][-1:]}
        )
        config = structured_writer.create_config(
            pattern=pattern,
            table="t",
            conditions=[structured_writer.Condition.is_end_episode()],
        )
        writer = client.structured_writer([config])
        writer.append({"obs": torch.tensor([1.0, 2.0])})
        writer.end_episode()
        writer.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]])

    def test_bfloat16_write_raises_clear_error(self):
        server = _make_server()
        client = server.in_process_client
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            with self.assertRaisesRegex(ValueError, "torch.bfloat16"):
                w.append({"obs": torch.ones(2, dtype=torch.bfloat16)})

    def test_cuda_tensor_write_matches_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        server = _make_server()
        client = server.in_process_client
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": torch.tensor([1.0, 2.0], device="cuda")})
            w.create_item("t", priority=1.0, trajectory={"obs": w.history["obs"][:]})
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]])

    def test_create_reference_step_and_infer_signature_accept_torch(self):
        step_spec = {
            "a": torch.zeros((), dtype=torch.float32),
            "b": torch.zeros([2, 2], dtype=torch.int32),
        }
        ref_step = structured_writer.create_reference_step(step_spec)
        # _RefNode 必须经 __getitem__ 才产出 PatternNode(参考 structured_writer_test)。
        pattern = {"a": ref_step["a"][-1:], "b": ref_step["b"][-2:]}
        config = structured_writer.create_config(
            pattern=pattern,
            table="t",
            conditions=[structured_writer.Condition.is_end_episode()],
        )
        signature = structured_writer.infer_signature([config], step_spec)
        self.assertEqual(signature["a"].dtype, np.float32)
        self.assertEqual(signature["b"].dtype, np.int32)
        self.assertEqual(tuple(signature["b"].shape), (2, 2, 2))


if __name__ == "__main__":
    absltest.main()
