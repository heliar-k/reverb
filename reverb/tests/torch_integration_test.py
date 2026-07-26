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


def _make_server(table_name="t", max_times_sampled=1, **server_kwargs):
    return reverb.Server(
        tables=[
            reverb.Table(
                name=table_name,
                sampler=reverb.selectors.Fifo(),
                remover=reverb.selectors.Fifo(),
                max_size=10,
                max_times_sampled=max_times_sampled,
                rate_limiter=reverb.rate_limiters.MinSize(1),
            )
        ],
        in_process=True,
        **server_kwargs,
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

    def test_legacy_writer_append_sequence_accepts_torch(self):
        server = _make_server()
        client = server.in_process_client
        with client.writer(max_sequence_length=2, chunk_length=2) as w:
            w.append_sequence({"obs": torch.tensor([[1.0], [2.0]])})
            w.create_item("t", num_timesteps=2, priority=1.0)
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0], [2.0]])

    def test_legacy_writer_append_sequence_bfloat16_raises(self):
        server = _make_server()
        client = server.in_process_client
        with (
            client.writer(max_sequence_length=2, chunk_length=2) as w,
            self.assertRaisesRegex(ValueError, "torch.bfloat16"),
        ):
            w.append_sequence({"obs": torch.ones((2, 1), dtype=torch.bfloat16)})

    def test_grpc_client_accepts_torch(self):
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
            ]
        )
        self.addCleanup(server.stop)
        client = reverb.Client(f"localhost:{server.port}")
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": torch.tensor([1.0, 2.0])})
            w.create_item("t", priority=1.0, trajectory={"obs": w.history["obs"][:]})
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]])

    def test_shm_client_accepts_torch(self):
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
            shm=True,
        )
        self.addCleanup(server.stop)
        client = reverb.ShmClient(server.shm_socket_path)
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": torch.tensor([1.0, 2.0])})
            w.create_item("t", priority=1.0, trajectory={"obs": w.history["obs"][:]})
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]])

    def test_output_format_uint16_converts_when_torch_supports(self):
        # torch>=2.3 has uint16/32/64: from_numpy converts (no fallback).
        # Older torch would fall back to numpy — both behaviors are valid;
        # this test pins values being correct either way.
        server = _make_server(output_format="torch")
        client = server.in_process_client
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"x": np.array([7], dtype=np.uint16)})
            w.create_item("t", priority=1.0, trajectory={"x": w.history["x"][:]})
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        leaf = samples[0].data[0]
        self.assertIsInstance(leaf, (np.ndarray, torch.Tensor))
        np.testing.assert_array_equal(np.asarray(leaf), [[7]])

    def test_bfloat16_write_raises_clear_error(self):
        server = _make_server()
        client = server.in_process_client
        with (
            client.trajectory_writer(num_keep_alive_refs=1) as w,
            self.assertRaisesRegex(ValueError, "torch.bfloat16"),
        ):
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

    def test_output_format_torch_returns_tensors(self):
        server = _make_server(max_times_sampled=0, output_format="torch")
        client = server.in_process_client
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": np.array([1.0, 2.0], dtype=np.float32)})
            w.create_item("t", priority=1.0, trajectory={"obs": w.history["obs"][:]})
            w.flush()

        # emit_timesteps=False: trajectory leaves are torch tensors.
        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertIsInstance(samples[0].data[0], torch.Tensor)
        np.testing.assert_allclose(samples[0].data[0].numpy(), [[1.0, 2.0]])

        # emit_timesteps=True (LocalClient default): timestep leaves too.
        draws = list(client.sample("t", num_samples=1))
        leaf = draws[0][0].data[0]
        self.assertIsInstance(leaf, torch.Tensor)
        np.testing.assert_allclose(leaf.numpy(), [1.0, 2.0])

    def test_output_format_torch_unsupported_dtype_falls_back(self):
        server = _make_server(output_format="torch")
        client = server.in_process_client
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"tag": np.array([b"ab"], dtype="S2")})
            w.create_item("t", priority=1.0, trajectory={"tag": w.history["tag"][:]})
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        leaf = samples[0].data[0]
        self.assertIsInstance(leaf, np.ndarray)
        self.assertEqual(leaf.dtype, np.dtype("S2"))
        self.assertEqual(leaf[0][0], b"ab")

    def test_output_format_invalid_raises_at_construction(self):
        with self.assertRaisesRegex(ValueError, "output_format"):
            _make_server(output_format="xml")
        with self.assertRaisesRegex(ValueError, "output_format"):
            reverb.Client("localhost:1", output_format="xml")
        # in_process=False must also reject (Spec FR2: 构造时即 ValueError)。
        with self.assertRaisesRegex(ValueError, "output_format"):
            reverb.Server(
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
                output_format="xml",
            )

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


class TorchUnavailableTest(absltest.TestCase):
    """Runs only when torch is NOT installed (i.e. the bazel test env)."""

    def setUp(self):
        super().setUp()
        if torch is not None:
            self.skipTest("only meaningful without torch")

    def test_torch_output_without_torch_raises_importerror(self):
        server = _make_server(output_format="torch")
        client = server.in_process_client
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": np.array([1.0], dtype=np.float32)})
            w.create_item("t", priority=1.0, trajectory={"obs": w.history["obs"][:]})
            w.flush()

        with self.assertRaises(ImportError):
            list(client.sample("t", num_samples=1, emit_timesteps=False))


if __name__ == "__main__":
    absltest.main()
