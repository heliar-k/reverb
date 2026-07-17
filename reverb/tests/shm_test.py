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
import threading
import time

import numpy as np
from absl.testing import absltest

import reverb
from reverb import errors, signature_codec, structured_writer


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

    def test_pickle_raises_pickling_error(self):
        server, client = _make_shm_server()
        # ShmClient.__reduce__ raises pickle.PicklingError specifically (not
        # the looser TypeError/ValueError union): the SHM mmap + ring state
        # can't survive a pickle round-trip. Reconnect by socket_path instead.
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(client)

    def test_pickle_error_message_guides_recovery(self):
        server, client = _make_shm_server()
        with self.assertRaisesRegex(
            pickle.PicklingError, r"Reconnect with ShmClient\(socket_path\)"
        ):
            pickle.dumps(client)


class ShmClientReprTest(absltest.TestCase):
    def test_repr_contains_socket_path(self):
        server, client = _make_shm_server()
        # repr mirrors Client/LocalClient: f"ShmClient(socket_path={path})".
        self.assertEqual(repr(client), f"ShmClient(socket_path={client._socket_path})")


class ShmClientServerInfoTest(absltest.TestCase):
    """ticket ⑧ step 1: server_info() returns a bootstrap-time snapshot of the
    server's TableInfo (piggybacked on the SHM handshake), not an empty {}.
    The snapshot reflects table state at Connect time; mid-session
    Table.replace / signature changes are NOT reflected (step 2 deferred)."""

    def test_server_info_returns_real_metadata(self):
        server, client = _make_shm_server(table_name="t", max_size=7)
        info = client.server_info()
        self.assertIsInstance(info, dict)
        self.assertIn("t", info)
        self.assertEqual(info["t"].max_size, 7)
        # sampler/remover options are populated (Fifo selectors by default).
        self.assertIsNotNone(info["t"].sampler_options)
        self.assertIsNotNone(info["t"].remover_options)
        # No signature declared on this table -> None.
        self.assertIsNone(info["t"].signature)
        # The snapshot is taken at Connect time, before any insert -> empty.
        self.assertEqual(info["t"].current_size, 0)

    def test_server_info_reflects_size_at_connect_time(self):
        # Insert items via the in-process path BEFORE the ShmClient connects,
        # so the bootstrap snapshot sees a non-empty table.
        server, _ = _make_shm_server(table_name="t", max_size=10, min_size=1)
        local = server.in_process_client
        for i in range(3):
            _insert_one(local, "t", np.array([float(i)], dtype=np.float32))

        # Now connect an ShmClient; its snapshot captures the 3 items.
        client = reverb.ShmClient(server.shm_socket_path)
        info = client.server_info()
        self.assertEqual(info["t"].current_size, 3)

    def test_server_info_carries_table_signature(self):
        # A table built with a signature propagates it through the bootstrap
        # snapshot so _get_signature_for_table (and thus
        # sample(unpack_as_table_signature=True)) can find the table.
        sig = {"v": signature_codec.TensorSpec((None, 1), np.float32, "v")}
        server = reverb.Server(
            tables=[
                reverb.Table(
                    name="t",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=10,
                    max_times_sampled=1,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    signature=sig,
                )
            ],
            in_process=True,
            shm=True,
        )
        client = reverb.ShmClient(server.shm_socket_path)
        info = client.server_info()
        self.assertIn("t", info)
        self.assertIsNotNone(info["t"].signature)
        # The signature cache is populated from server_info().
        self.assertIn("t", client._signature_cache)

    def test_sample_unpack_as_table_signature_works(self):
        # The regression that motivated ticket ⑧: before the fix,
        # sample(unpack_as_table_signature=True) threw
        # `ValueError: Could not find table` because server_info() returned {}
        # and the signature cache was empty. Now the bootstrap snapshot feeds
        # the cache, so unpacking against the table signature works.
        sig = {
            "obs": signature_codec.TensorSpec((None, 1), np.float32, "obs"),
        }
        server = reverb.Server(
            tables=[
                reverb.Table(
                    name="t",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=10,
                    max_times_sampled=1,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    signature=sig,
                )
            ],
            in_process=True,
            shm=True,
        )
        client = reverb.ShmClient(server.shm_socket_path)
        with client.trajectory_writer(num_keep_alive_refs=3) as w:
            for i in range(3):
                w.append({"obs": np.array([float(i)], dtype=np.float32)})
            w.create_item(
                table="t",
                priority=1.0,
                trajectory={"obs": w.history["obs"][:]},
            )
            w.flush()

        sample = next(
            client.sample(
                "t",
                num_samples=1,
                unpack_as_table_signature=True,
                emit_timesteps=False,
            )
        )
        # With a table signature + unpack_as_table_signature=True, data is a
        # dict keyed by the signature's column names (not a flat list).
        self.assertIsInstance(sample.data, dict)
        self.assertEqual(set(sample.data.keys()), {"obs"})
        np.testing.assert_array_equal(sample.data["obs"], [[0.0], [1.0], [2.0]])

    def test_server_info_accepts_timeout_kwarg(self):
        # The hook accepts `timeout` for parity with the gRPC/Local hooks;
        # it is ignored (no round-trip — the data is cached at Connect).
        server, client = _make_shm_server()
        info = client.server_info(timeout=1)
        self.assertIn("t", info)


class ShmConcurrentWriterSamplerTest(absltest.TestCase):
    """Decision D regression: a writer worker thread and a sampler worker
    thread on ONE ShmClient must run concurrently without corrupting the
    shared transport.

    Pre-D the client had a single SPSC c2s/s2c ring pair, and both
    TrajectoryWriter (RunShmWorker) and ShmSampler start background worker
    threads that wrote that ring as producers. Two producers on one SPSC
    `head` (no CAS) corrupted the ring and hung both workers in
    ReadBlocking. Decision D gives each flow its own ring pair, restoring the
    single-producer invariant. This test would hang/corrupt before D.
    """

    def test_concurrent_writer_and_sampler_on_one_client(self):
        server, client = _make_shm_server(table_name="t", max_size=10000, min_size=1)
        stop = threading.Event()
        errs = []
        items_written = []
        samples_seen = []

        def writer_loop():
            try:
                i = 0
                while not stop.is_set():
                    # One writer per item (mirrors the passing write/sample
                    # tests' with-block). Reusing a single writer across many
                    # create_item calls leaves rolled-off None refs in
                    # history[:], which CreateItem rejects; a fresh writer per
                    # item sidesteps that. The point is concurrent transport
                    # progress, not writer reuse throughput.
                    with client.trajectory_writer(num_keep_alive_refs=1) as w:
                        w.append({"obs": np.array([float(i)], dtype=np.float32)})
                        w.create_item(
                            table="t",
                            priority=1.0,
                            trajectory={"obs": w.history["obs"][:]},
                        )
                        w.flush()
                    items_written.append(i)
                    i += 1
            except Exception as e:  # noqa: BLE001
                errs.append(("writer", repr(e)))

        def sampler_loop():
            try:
                # A short rate-limiter timeout keeps the sampler from blocking
                # forever when the writer hasn't flushed yet; a timeout is a
                # normal "no item right now" signal, so we retry. The point of
                # the test is that the sampler's worker thread and the writer's
                # worker thread make concurrent progress WITHOUT corrupting the
                # transport — not that a sample is always available.
                while not stop.is_set():
                    try:
                        for s in client.sample(
                            "t",
                            num_samples=1,
                            emit_timesteps=False,
                            timeout_ms=200,
                        ):
                            samples_seen.append(s)
                            break
                    except errors.DeadlineExceededError:
                        continue  # table momentarily empty; retry
            except Exception as e:  # noqa: BLE001
                errs.append(("sampler", repr(e)))

        t_w = threading.Thread(target=writer_loop)
        t_s = threading.Thread(target=sampler_loop)
        t_w.start()
        t_s.start()
        # Run concurrently for a few seconds. Pre-D this hangs inside the
        # first flush/sample (two producers on one SPSC ring); post-D both
        # threads make progress.
        time.sleep(3.0)
        stop.set()
        t_w.join(timeout=15)
        t_s.join(timeout=15)

        # Both threads must have exited (not hung on a corrupted ring).
        self.assertFalse(t_w.is_alive(), "writer thread hung")
        self.assertFalse(t_s.is_alive(), "sampler thread hung")
        # No errors surfaced from either worker.
        self.assertEqual(errs, [])
        # And real work happened in BOTH flows concurrently.
        self.assertGreater(len(items_written), 0, "writer made no progress")
        self.assertGreater(len(samples_seen), 0, "sampler made no progress")

        server.stop()


class ShmMultiTableTest(absltest.TestCase):
    """ticket ⑨: an ShmServer holds ALL tables and routes by table name.

    Pre-⑨ the server held exactly ONE table (tables[0]); a multi-table
    `Server(shm=True)` silently dropped items targeting any other table and
    could not sample them. These tests pin the routing on both the sample and
    insert paths, the server_info snapshot, and NOT_FOUND surfacing for an
    unknown table.
    """

    def _make_two_table_server(self):
        """uniform_table (Uniform) + fifo_table (Fifo), both MinSize(1)."""
        uniform_table = reverb.Table(
            name="uniform_table",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )
        fifo_table = reverb.Table(
            name="fifo_table",
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )
        server = reverb.Server(
            tables=[uniform_table, fifo_table],
            in_process=True,
            shm=True,
        )
        return server

    def test_multi_table_server_info_lists_all_tables(self):
        server = self._make_two_table_server()
        client = reverb.ShmClient(server.shm_socket_path)
        info = client.server_info()
        self.assertEqual(set(info.keys()), {"uniform_table", "fifo_table"})
        self.assertEqual(info["uniform_table"].max_size, 10)
        self.assertEqual(info["fifo_table"].max_size, 10)
        server.stop()

    def test_multi_table_sample_routes_by_name(self):
        # Pre-seed both tables via the in-process client (so data exists before
        # SHM sampling), then sample each back over SHM by table name and
        # verify no cross-contamination: uniform_table holds only 0.0 and
        # fifo_table holds only 1.0.
        server = self._make_two_table_server()
        local = server.in_process_client
        _insert_one(local, "uniform_table", np.array([0.0], dtype=np.float32))
        _insert_one(local, "fifo_table", np.array([1.0], dtype=np.float32))

        client = reverb.ShmClient(server.shm_socket_path)
        u = next(client.sample("uniform_table", num_samples=1, emit_timesteps=False))
        f = next(client.sample("fifo_table", num_samples=1, emit_timesteps=False))
        np.testing.assert_array_equal(u.data[0], [[0.0]])
        np.testing.assert_array_equal(f.data[0], [[1.0]])
        server.stop()

    def test_multi_table_insert_routes_by_name(self):
        # Insert into each table over SHM via trajectory_writer, then sample
        # each back and verify routing (distinct values per table).
        server = self._make_two_table_server()
        client = reverb.ShmClient(server.shm_socket_path)
        _insert_one(client, "uniform_table", np.array([10.0], dtype=np.float32))
        _insert_one(client, "fifo_table", np.array([20.0], dtype=np.float32))

        u = next(client.sample("uniform_table", num_samples=1, emit_timesteps=False))
        f = next(client.sample("fifo_table", num_samples=1, emit_timesteps=False))
        np.testing.assert_array_equal(u.data[0], [[10.0]])
        np.testing.assert_array_equal(f.data[0], [[20.0]])
        server.stop()

    def test_unknown_table_sample_returns_error(self):
        # An unknown table name on the sample path surfaces as FileNotFoundError
        # (absl::kNotFound -> PyExc_FileNotFoundError via MaybeRaiseFromStatus).
        server = self._make_two_table_server()
        client = reverb.ShmClient(server.shm_socket_path)
        with self.assertRaises(FileNotFoundError):
            next(client.sample("nonexistent", num_samples=1, emit_timesteps=False))
        server.stop()

    def test_unknown_table_insert_returns_error(self):
        # An unknown table name on the insert path surfaces as FileNotFoundError
        # too (RunShmWorker maps ShmError::NOT_FOUND -> absl::NotFoundError,
        # raised on flush()).
        server = self._make_two_table_server()
        client = reverb.ShmClient(server.shm_socket_path)
        with self.assertRaises(FileNotFoundError):
            _insert_one(client, "nonexistent", np.array([0.0], dtype=np.float32))
        server.stop()


class ShmMutateResetTest(absltest.TestCase):
    """ticket ⑩: mutate_priorities / reset over the SHM transport.

    These ride the INSERT flow (insert_c2s/insert_s2c) under
    ShmConnection::insert_flow_mu so they never race RunShmWorker as a second
    producer on the insert ring. Mirror in_process_test.py's
    test_in_process_mutate_and_reset and client_test.py's mutate/reset cases.
    """

    def _make_server(self, table_name="t", max_size=10, max_times_sampled=0):
        # max_times_sampled=0 keeps sampled items reusable so a second sample
        # after a mutate does not block on an empty table (MinSize(1)).
        return _make_shm_server(
            table_name=table_name,
            max_size=max_size,
            min_size=1,
            max_times_sampled=max_times_sampled,
        )

    def test_mutate_priorities_updates_priority(self):
        server, client = self._make_server()
        _insert_one(client, "t", np.array([1.0], dtype=np.float32))
        key = next(client.sample("t", num_samples=1, emit_timesteps=False)).info.key
        client.mutate_priorities("t", updates={key: 42.5})
        after = next(
            client.sample("t", num_samples=1, emit_timesteps=False)
        ).info.priority
        self.assertAlmostEqual(after, 42.5)
        server.stop()

    def test_mutate_priorities_deletes_item(self):
        # server_info() is a bootstrap snapshot (won't reflect the delete), so
        # verify via the live in-process client's server_info, which sees the
        # real table state.
        server, client = self._make_server(max_size=10)
        for i in range(3):
            _insert_one(client, "t", np.array([float(i)], dtype=np.float32))
        self.assertEqual(server.in_process_client.server_info()["t"].current_size, 3)
        key = next(client.sample("t", num_samples=1, emit_timesteps=False)).info.key
        client.mutate_priorities("t", deletes=[key])
        self.assertEqual(server.in_process_client.server_info()["t"].current_size, 2)
        server.stop()

    def test_reset_clears_table(self):
        server, client = self._make_server(max_size=10)
        for i in range(3):
            _insert_one(client, "t", np.array([float(i)], dtype=np.float32))
        self.assertEqual(server.in_process_client.server_info()["t"].current_size, 3)
        client.reset("t")
        # The live in-process client sees the post-reset state (current_size 0).
        self.assertEqual(server.in_process_client.server_info()["t"].current_size, 0)
        # And sampling an empty table with a short rate-limiter timeout raises
        # DeadlineExceededError (MinSize(1) blocks when the table is empty).
        with self.assertRaises(errors.DeadlineExceededError):
            next(
                client.sample("t", num_samples=1, timeout_ms=300, emit_timesteps=False)
            )
        server.stop()

    def test_mutate_priorities_unknown_table_raises(self):
        # absl::NotFoundError -> Python FileNotFoundError (ticket ⑨ mapping).
        server, client = self._make_server()
        with self.assertRaises(FileNotFoundError):
            client.mutate_priorities("nonexistent", updates={1: 1.0})
        server.stop()

    def test_reset_unknown_table_raises(self):
        server, client = self._make_server()
        with self.assertRaises(FileNotFoundError):
            client.reset("nonexistent")
        server.stop()

    def test_mutate_reset_does_not_corrupt_concurrent_inserts(self):
        # Validates the insert_flow_mu: a control-plane call (mutate_priorities)
        # on the caller thread overlaps RunShmWorker inserts on the writer's
        # background thread — both touch insert_c2s. The mutex serializes them
        # so neither corrupts the ring. Run inserts in a background thread,
        # fire mutate_priorities on the main thread, then verify every inserted
        # item samples back intact (no corruption / lost items).
        server, client = self._make_server(table_name="t", max_size=500)
        stop = threading.Event()
        errs = []
        written = []

        def writer_loop():
            try:
                i = 0
                while not stop.is_set():
                    val = float(i)
                    with client.trajectory_writer(num_keep_alive_refs=1) as w:
                        w.append({"v": np.array([val], dtype=np.float32)})
                        w.create_item(
                            table="t",
                            priority=1.0,
                            trajectory={"v": w.history["v"][:]},
                        )
                        w.flush()
                    written.append(val)
                    i += 1
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

        t_w = threading.Thread(target=writer_loop)
        t_w.start()
        # While the writer thread hammers insert_c2s, repeatedly mutate a
        # (possibly absent) key's priority on the main thread. This is the
        # control-plane-vs-insert overlap the mutex protects. MutateItems
        # ignores absent keys, so this is a safe no-op on data.
        deadline = time.time() + 2.0
        mutate_calls = 0
        while time.time() < deadline:
            client.mutate_priorities("t", updates={0: 1.0})
            mutate_calls += 1
        stop.set()
        t_w.join(timeout=15)
        self.assertFalse(t_w.is_alive(), "writer thread hung")
        self.assertEqual(errs, [])
        self.assertGreater(mutate_calls, 0, "no mutate calls made")
        self.assertGreater(len(written), 0, "writer made no progress")
        # Drain samples and confirm the values are an intact subset of what was
        # written (no corruption: each sampled value must be one we wrote).
        written_set = set(written)
        seen = set()
        try:
            for s in client.sample(
                "t", num_samples=200, emit_timesteps=False, timeout_ms=200
            ):
                seen.add(float(np.asarray(s.data[0]).reshape(-1)[0]))
        except errors.DeadlineExceededError:
            pass  # table drained / rate-limiter timeout — fine
        self.assertTrue(seen.issubset(written_set), f"corruption: {seen - written_set}")
        server.stop()


if __name__ == "__main__":
    absltest.main()
