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

"""End-to-end smoke test for the gRPC client/server path (no TensorFlow).

Mirrors `in_process_test.py` but exercises the real gRPC `Client` against a
networked `Server(in_process=False)`. Migrated to absltest from the historical
bare-assert + __main__ form.
"""

import numpy as np
import portpicker
from absl.testing import absltest

import reverb


def _make_grpc_server(table_name="t", max_size=10, min_size=1):
    port = portpicker.pick_unused_port()
    server = reverb.Server(
        tables=[
            reverb.Table(
                name=table_name,
                sampler=reverb.selectors.Fifo(),
                remover=reverb.selectors.Fifo(),
                max_size=max_size,
                rate_limiter=reverb.rate_limiters.MinSize(min_size),
            )
        ],
        port=port,
        in_process=False,
    )
    return server, port


class GrpcClientServerInfoTest(absltest.TestCase):
    def test_grpc_client_server_info(self):
        server, port = _make_grpc_server()
        try:
            client = reverb.Client(f"localhost:{port}")
            info = client.server_info()
            self.assertIn("t", info)
            self.assertEqual(info["t"].max_size, 10)
        finally:
            server.stop()


class GrpcWriterInsertAndSampleTest(absltest.TestCase):
    def test_grpc_writer_insert_and_sample(self):
        server, port = _make_grpc_server(table_name="q", max_size=10, min_size=1)
        try:
            client = reverb.Client(f"localhost:{port}")
            with client.writer(max_sequence_length=2) as w:
                w.append({"obs": np.array([1.0, 2.0], dtype=np.float32)})
                w.append({"obs": np.array([3.0, 4.0], dtype=np.float32)})
                w.create_item(table="q", num_timesteps=1, priority=1.0)
                w.flush()

            samples = list(client.sample("q", num_samples=1))
            self.assertLen(samples, 1)
            timesteps = samples[0]
            self.assertLen(timesteps, 1)
            data = timesteps[0].data
            np.testing.assert_allclose(np.asarray(data[0]), [3.0, 4.0])
        finally:
            server.stop()


class GrpcMutateAndResetTest(absltest.TestCase):
    def test_grpc_mutate_priorities_and_reset(self):
        server, port = _make_grpc_server()
        try:
            client = reverb.Client(f"localhost:{port}")
            with client.writer(max_sequence_length=1) as w:
                w.append({"v": np.array(7, dtype=np.int64)})
                w.create_item(table="t", num_timesteps=1, priority=1.0)
                w.flush()

            info = client.server_info()
            self.assertEqual(info["t"].current_size, 1)

            client.reset("t")
            info_after = client.server_info()
            self.assertEqual(info_after["t"].current_size, 0)
        finally:
            server.stop()


if __name__ == "__main__":
    absltest.main()
