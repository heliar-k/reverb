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

"""End-to-end test for the SHM (shared-memory) Reverb transport (ticket ⑤).

Exercises the full Server(shm=True) -> ShmClient -> trajectory_writer/
structured_writer -> sample round trip over POSIX shared memory, mirroring
`in_process_test.py` for the third transport. No TensorFlow; numpy only.

v1 SHM scope (per spec §6 + ticket ⑤):
  - ONE table per ShmServer (// ponytail: multi-table later).
  - No MutatePriorities/Reset/Checkpoint/ServerInfo round-trip over SHM
    (the C++ ShmServer only handles SAMPLE/RELEASE/INSERT/ALLOCATE). These are
    therefore NOT exercised here. `server_info` returns a stub (// ponytail:).
"""

import os
import pickle
import tempfile

from absl.testing import absltest
import numpy as np

import reverb
from reverb import errors
from reverb import structured_writer


def _make_table(
    table_name="t",
    max_size=10,
    min_size=1,
    sampler=None,
    remover=None,
    max_times_sampled=1,
):
    """Single-table definition for an SHM server."""
    return reverb.Table(
        name=table_name,
        sampler=sampler or reverb.selectors.Fifo(),
        remover=remover or reverb.selectors.Fifo(),
        max_size=max_size,
        max_times_sampled=max_times_sampled,
        rate_limiter=reverb.rate_limiters.MinSize(min_size),
    )


def _make_shm_server(
    table_name="t",
    max_size=10,
    min_size=1,
    sampler=None,
    remover=None,
    max_times_sampled=1,
    **kwargs,
):
    """Builds a single-table SHM server + a connected ShmClient.

    v1: ShmServer holds ONE table. The server owns the table directly via
    `in_process=True` (no gRPC port) and additionally exposes it over the SHM
    transport via `shm=True`. SHM is an additional surface layered on the owned
    Table.
    """
    table = _make_table(
        table_name=table_name,
        max_size=max_size,
        min_size=min_size,
        sampler=sampler,
        remover=remover,
        max_times_sampled=max_times_sampled,
    )
    # in_process=True owns the Table without gRPC; shm=True layers the SHM
    # transport on it. Callers may override via kwargs (e.g. shm_socket_path).
    server = reverb.Server(tables=[table], in_process=True, shm=True, **kwargs)
    client = reverb.ShmClient(server.shm_socket_path)
    return server, client


def _insert_one(client, table, value, priority=1.0, num_keep_alive_refs=1):
    """Inserts a single-item trajectory carrying one scalar-bearing column."""
    with client.trajectory_writer(num_keep_alive_refs=num_keep_alive_refs) as w:
        w.append({"v": np.asarray(value)})
        w.create_item(
            table=table, priority=priority, trajectory={"v": w.history["v"][:]}
        )
        w.flush()


class ShmWriteSampleTest(absltest.TestCase):
    """Core ⑤ round trip: trajectory_writer over SHM -> sample reads back."""

    def test_shm_write_sample(self):
        server, client = _make_shm_server()
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": np.array([1.0, 2.0], dtype=np.float32)})
            w.create_item(
                table="t", priority=1.0, trajectory={"obs": w.history["obs"][:]}
            )
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]])

    def test_shm_multiple_steps(self):
        server, client = _make_shm_server(table_name="q")
        with client.trajectory_writer(num_keep_alive_refs=3) as w:
            for i in range(3):
                w.append({"obs": np.array([float(i)], dtype=np.float32)})
            w.create_item(
                table="q", priority=1.0, trajectory={"obs": w.history["obs"][:]}
            )
            w.flush()

        samples = list(client.sample("q", num_samples=1, emit_timesteps=False))
        np.testing.assert_allclose(
            np.asarray(samples[0].data[0]), [[0.0], [1.0], [2.0]]
        )

    def test_shm_multi_chunk_item(self):
        # max_chunk_length=2 over 5 steps -> chunks [2,2,1]; sampler concatenates.
        server, client = _make_shm_server()
        with client.trajectory_writer(num_keep_alive_refs=5, max_chunk_length=2) as w:
            for i in range(5):
                w.append({"v": np.array([float(i)], dtype=np.float32)})
            w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(
            np.asarray(samples[0].data[0]).reshape(-1), [0.0, 1.0, 2.0, 3.0, 4.0]
        )


class ShmStructuredWriterTest(absltest.TestCase):
    """StructuredWriter over SHM -> sample reads back."""

    def test_trajectory_pattern(self):
        server, client = _make_shm_server(table_name="sw", max_size=50, min_size=1)

        step_spec = {"a": np.zeros([], np.float32)}
        ref_step = structured_writer.create_reference_step(step_spec)
        pattern = {"x": ref_step["a"][-3:]}  # last 3 steps of column 'a'
        config = structured_writer.create_config(pattern=pattern, table="sw")

        writer = client.structured_writer(configs=[config])
        for i in range(3):
            writer.append(np.asarray(float(i), dtype=np.float32))
        writer.end_episode()

        samples = list(client.sample("sw", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_array_equal(np.asarray(samples[0].data[0]), [0.0, 1.0, 2.0])


class ShmSelectorTest(absltest.TestCase):
    """Sampler strategies over SHM."""

    def test_lifo_replay(self):
        server, client = _make_shm_server(
            table_name="l",
            max_size=10,
            min_size=1,
            sampler=reverb.selectors.Lifo(),
            max_times_sampled=1,
        )
        for i in range(4):
            _insert_one(client, "l", np.array([float(i)], dtype=np.float32))

        order = [
            float(np.asarray(sample.data[0]).reshape(-1)[0])
            for sample in client.sample("l", num_samples=4, emit_timesteps=False)
        ]
        self.assertEqual(order, [3.0, 2.0, 1.0, 0.0])


class ShmDtypesTest(absltest.TestCase):
    """dtype fidelity across the SHM transport."""

    def test_multiple_dtypes(self):
        server, client = _make_shm_server(
            table_name="d", max_size=10, min_size=1, max_times_sampled=1
        )
        cases = [
            np.array([3.14], dtype=np.float32),
            np.array([2.71], dtype=np.float64),
            np.array([-7], dtype=np.int32),
            np.array([2**40], dtype=np.int64),
            np.array([255], dtype=np.uint8),
            np.array([True], dtype=np.bool_),
        ]
        for arr in cases:
            _insert_one(client, "d", arr)

        samples = list(client.sample("d", num_samples=len(cases), emit_timesteps=False))
        self.assertLen(samples, len(cases))
        for arr, sample in zip(cases, samples):
            got = np.asarray(sample.data[0]).reshape(-1)[0]
            self.assertEqual(got.dtype, arr.dtype, (got.dtype, arr.dtype))
            exp = arr[0]
            if np.issubdtype(arr.dtype, np.bool_):
                self.assertEqual(bool(got), bool(exp))
            elif np.issubdtype(arr.dtype, np.floating):
                self.assertTrue(np.isclose(got, exp))
            else:
                self.assertEqual(int(got), int(exp))


class ShmRateLimiterTest(absltest.TestCase):
    """Rate limiter timeout surfaces as DeadlineExceededError over SHM."""

    def test_rate_limiter_timeout(self):
        server, client = _make_shm_server(
            table_name="r", max_size=20, min_size=5, sampler=reverb.selectors.Fifo()
        )
        for i in range(2):
            _insert_one(client, "r", np.array([float(i)], dtype=np.float32))

        import time

        start = time.time()
        with self.assertRaises(errors.DeadlineExceededError):
            list(
                client.sample("r", num_samples=1, timeout_ms=500, emit_timesteps=False)
            )
        elapsed = time.time() - start
        self.assertLess(elapsed, 5.0, f"timeout took too long: {elapsed:.1f}s")


class ShmServerLifecycleTest(absltest.TestCase):
    """C1: ShmServer lifecycle hangs off the Server object."""

    def test_shm_socket_path_exposed(self):
        server, _ = _make_shm_server()
        path = server.shm_socket_path
        self.assertIsInstance(path, str)
        self.assertTrue(path, "socket path must be non-empty")
        # The udsocket file must exist while the server is running.
        self.assertTrue(os.path.exists(path), path)

    def test_custom_socket_path(self):
        custom = os.path.join(tempfile.mkdtemp(), "my_shm.sock")
        server = reverb.Server(tables=[_make_table()], shm=True, shm_socket_path=custom)
        self.assertEqual(server.shm_socket_path, custom)

    def test_stop_cleans_up_socket(self):
        server, _ = _make_shm_server()
        path = server.shm_socket_path
        self.assertTrue(os.path.exists(path))
        server.stop()
        # The bootstrap udsocket must be unlinked on Stop (R7/C1).
        self.assertFalse(os.path.exists(path), f"socket not cleaned up: {path}")


class ShmClientNotPicklableTest(absltest.TestCase):
    """R13: ShmClient holds SHM mmap + ring state -> not picklable."""

    def test_pickle_raises(self):
        server, client = _make_shm_server()
        with self.assertRaises((pickle.PicklingError, TypeError, ValueError)):
            pickle.dumps(client)


class ShmClientReprTest(absltest.TestCase):
    def test_repr(self):
        server, client = _make_shm_server()
        s = repr(client)
        self.assertIn("ShmClient", s)


if __name__ == "__main__":
    absltest.main()
