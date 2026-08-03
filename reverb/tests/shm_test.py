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

Coverage spans v1 (tickets ①-⑦: ring/bootstrap/pool/sample/insert/crash)
and v2 (⑧ server_info on-demand round-trip, ⑨ multi-table, ⑩ mutate/reset +
deadlock regression, ⑫ pickle, ⑬ legacy writer/insert NotImplementedError,
⑧-2b validate_items, ⑪ checkpoint). The C++ ShmServer dispatch handles
SAMPLE/RELEASE/INSERT/ALLOCATE plus the control-plane ops mutate_priorities/
reset/checkpoint (all riding the insert flow under a client mutex).
"""

import gc
import os
import pickle
import tempfile
import threading
import time
import weakref

import numpy as np
from absl.testing import absltest

import reverb
from reverb import errors, signature_codec, structured_writer
from reverb.platform.default import checkpointers


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

    The server owns the table directly via `in_process=True` (no gRPC port)
    and additionally exposes it over the SHM transport via `shm=True`. SHM is
    an additional surface layered on the owned Table (ShmServer supports all
    tables via name routing, ticket ⑨; this helper builds one for brevity).
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


class ShmPoolSlabConfigTest(absltest.TestCase):
    """ticket 02: SHM pool slab geometry is configurable from Server(...).

    shm_pool_slab_sizes / shm_pool_blocks_per_slab pass through
    Server -> pybind ShmServer -> ShmServer::Create -> ShmBytePool::Create.
    """

    def test_custom_geometry_roundtrip(self):
        # Small custom pool: 3 tiers, 8 blocks each (~533KB capacity vs the
        # ~1.4GB default). Small payloads insert/sample unchanged.
        server, client = _make_shm_server(
            shm_pool_slab_sizes=[64, 1024, 65536],
            shm_pool_blocks_per_slab=8,
        )
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": np.array([1.0, 2.0], dtype=np.float32)})
            w.create_item(
                table="t", priority=1.0, trajectory={"obs": w.history["obs"][:]}
            )
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]])
        server.stop()

    def test_invalid_geometry_raises_at_construction(self):
        # Non-ascending tiers: ShmBytePool::Create rejects with InvalidArgument
        # -> ValueError, before any SHM segment is created.
        with self.assertRaisesRegex(ValueError, "ascending"):
            _make_shm_server(shm_pool_slab_sizes=[1024, 64])

    def test_oversize_insert_surfaces_client_error(self):
        # A chunk bigger than the largest slab can never fit the pool: the
        # client must get a clear error (not hang/crash). Chain:
        # Allocate InvalidArgument -> ShmError::INVALID_ARGUMENT ->
        # RunShmWorker fail_stream -> flush raises ValueError.
        server, client = _make_shm_server(
            shm_pool_slab_sizes=[1024],
            shm_pool_blocks_per_slab=8,
        )
        # The chunker compresses, so zeros would shrink under the slab size —
        # use deterministic noise to keep the wire size above it.
        rng = np.random.RandomState(42)
        with self.assertRaisesRegex(ValueError, "too large"):
            _insert_one(client, "t", rng.randint(0, 256, size=4096, dtype=np.uint8))
        server.stop()

    def test_oversize_sample_surfaces_client_error(self):
        # Sample-path oversize: insert a big item via the in-process client
        # (bypasses the pool), then sample it over SHM — the result bytes
        # exceed the largest slab, so the server replies ERROR and the client
        # raises (RuntimeError from ShmError::INTERNAL) instead of hanging.
        server, client = _make_shm_server(
            shm_pool_slab_sizes=[1024],
            shm_pool_blocks_per_slab=8,
        )
        rng = np.random.RandomState(42)
        _insert_one(
            server.in_process_client,
            "t",
            rng.randint(0, 256, size=4096, dtype=np.uint8),
        )
        with self.assertRaises(RuntimeError):
            next(client.sample("t", num_samples=1, emit_timesteps=False))
        server.stop()


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


class ShmTwoServersOneProcessTest(absltest.TestCase):
    """Two Server(shm=True) in one process must not clobber each other.

    Regression: the default socket path was /tmp/reverb_shm_<pid>.sock and the
    C++ pool segment name was keyed by PID only, so starting a second server
    unlinked the FIRST server's live socket/pool (scan #12, server.py:421,
    shm_server.cc / bootstrap.cc).
    """

    def test_two_servers_distinct_and_both_usable(self):
        s1 = reverb.Server(tables=[_make_table("t")], in_process=True, shm=True)
        s2 = reverb.Server(tables=[_make_table("t")], in_process=True, shm=True)
        try:
            self.assertNotEqual(s1.shm_socket_path, s2.shm_socket_path)
            # Both must accept NEW connections after both are up (the second
            # server's bootstrap must not have unlinked the first's socket).
            c1 = reverb.ShmClient(s1.shm_socket_path)
            c2 = reverb.ShmClient(s2.shm_socket_path)
            _insert_one(c1, "t", np.array([1.0], dtype=np.float32))
            _insert_one(c2, "t", np.array([2.0], dtype=np.float32))
            v1 = float(
                np.asarray(
                    next(c1.sample("t", 1, emit_timesteps=False)).data[0]
                ).reshape(-1)[0]
            )
            v2 = float(
                np.asarray(
                    next(c2.sample("t", 1, emit_timesteps=False)).data[0]
                ).reshape(-1)[0]
            )
            self.assertEqual((v1, v2), (1.0, 2.0))
        finally:
            s1.stop()
            s2.stop()


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


class ShmClientPicklableTest(absltest.TestCase):
    """ticket ⑫: ShmClient pickles by socket_path and re-connects on unpickle.

    __reduce__ returns (ShmClient, (socket_path,)); unpickle calls __init__ →
    ShmClient::Connect (fresh bootstrap + mmap). The original client's mmap/ring
    state stays in the pickling process. No cross-process leak.
    """

    def test_pickle_roundtrip_reconnects_by_socket_path(self):
        server, client = _make_shm_server()
        _insert_one(client, "t", 1.0)
        client2 = pickle.loads(pickle.dumps(client))
        # New connection: a fresh ShmClient sharing the same server.
        self.assertIsInstance(client2, reverb.ShmClient)
        self.assertEqual(client2._socket_path, client._socket_path)
        # Original server still serves both clients.
        samples = list(client2.sample("t", num_samples=1, emit_timesteps=False))
        self.assertEqual(len(samples), 1)

    def test_pickled_client_can_insert_and_sample(self):
        server, client = _make_shm_server()
        client2 = pickle.loads(pickle.dumps(client))
        _insert_one(client2, "t", 42.0)
        # The inserted item is visible to the original client too (same table).
        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        np.testing.assert_array_equal(
            np.asarray(samples[0].data[0]).reshape(-1), [42.0]
        )

    def test_original_client_still_usable_after_pickle(self):
        server, client = _make_shm_server()
        _ = pickle.dumps(client)
        # Pickling does not tear down the source client's connection.
        _insert_one(client, "t", 7.0)
        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        np.testing.assert_array_equal(np.asarray(samples[0].data[0]).reshape(-1), [7.0])


class ShmClientLegacyWriterInsertNotImplementedTest(absltest.TestCase):
    """ticket ⑬: ShmClient raises clear NotImplementedError for legacy
    writer/insert (no SHM seam for the plain Writer) instead of reaching the
    C++ UnimplementedError at runtime."""

    def test_writer_raises_not_implemented(self):
        server, client = _make_shm_server()
        with self.assertRaisesRegex(
            NotImplementedError, r"trajectory_writer or structured_writer"
        ):
            client.writer(max_sequence_length=1)

    def test_insert_raises_not_implemented(self):
        server, client = _make_shm_server()
        with self.assertRaisesRegex(
            NotImplementedError, r"trajectory_writer or structured_writer"
        ):
            client.insert(np.asarray(1.0), {"t": 1.0})

    def test_trajectory_writer_still_works(self):
        # The supported path is unaffected by the legacy overrides.
        server, client = _make_shm_server()
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"v": np.asarray(1.0)})
            w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
            w.flush()


class ShmClientReprTest(absltest.TestCase):
    def test_repr_contains_socket_path(self):
        server, client = _make_shm_server()
        # repr mirrors Client/LocalClient: f"ShmClient(socket_path={path})".
        self.assertEqual(repr(client), f"ShmClient(socket_path={client._socket_path})")


class ShmClientLifetimeTest(absltest.TestCase):
    """Regression for the client-lifetime UAF (concurrency audit finding P0).

    The pybind ShmSampler/TrajectoryWriter/StructuredWriter borrow the
    ShmClient's connection (conn_) but previously nothing kept the client
    alive: the six factory defs had no py::keep_alive, and client.py adds no
    back-ref. A temporary client (`ShmClient(sock).trajectory_writer(...)`) or
    `del client` left the writer/sampler with a dangling connection — the
    ~ShmClient unmaps the rings/pool, so the next use is a genuine UAF. Fixed
    via py::keep_alive<0, 1> on all six factory defs in pybind.cc; these tests
    pin the behavior (pre-fix they UAF/segfault, e.g. under ASAN).
    """

    def test_keep_alive_pins_pybind_client(self):
        # Deterministic pin of the fix itself (not just its symptom): the
        # pybind ShmClient object must survive `del client` while a writer
        # born from it lives. Pre-fix the weakref dies immediately — no
        # reliance on the UAF actually crashing.
        server, client = _make_shm_server()
        client_ref = weakref.ref(client._client)
        writer = client.trajectory_writer(num_keep_alive_refs=1)
        del client
        gc.collect()
        self.assertIsNotNone(
            client_ref(),
            "pybind ShmClient died while a writer still borrows its connection",
        )
        del writer
        gc.collect()
        # Once the writer is gone the keep-alive lapses and the client (and
        # its connection) is reclaimed — no leak in the other direction.
        self.assertIsNone(client_ref(), "pybind ShmClient leaked past its writer")
        server.stop()

    def test_writer_outlives_temporary_client(self):
        # The temporary ShmClient wrapper dies at the end of the expression;
        # the writer born from it must still flush through a live connection.
        server = reverb.Server(tables=[_make_table("t")], in_process=True, shm=True)
        writer = reverb.ShmClient(server.shm_socket_path).trajectory_writer(
            num_keep_alive_refs=1
        )
        gc.collect()  # ensure the temporary wrapper is really gone
        with writer as w:
            w.append({"v": np.asarray(1.0)})
            w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
            w.flush()
        # The item really landed in the table (readable via a fresh client).
        client = reverb.ShmClient(server.shm_socket_path)
        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_array_equal(np.asarray(samples[0].data[0]).reshape(-1), [1.0])
        server.stop()

    def test_writer_outlives_deleted_client(self):
        server, client = _make_shm_server()
        writer = client.trajectory_writer(num_keep_alive_refs=1)
        del client
        gc.collect()
        with writer as w:
            w.append({"v": np.asarray(2.0)})
            w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
            w.flush()
        client2 = reverb.ShmClient(server.shm_socket_path)
        samples = list(client2.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_array_equal(np.asarray(samples[0].data[0]).reshape(-1), [2.0])
        server.stop()

    def test_sampler_outlives_deleted_client(self):
        server, client = _make_shm_server()
        _insert_one(client, "t", np.array([3.0], dtype=np.float32))
        # Grab the pybind sampler directly (bypassing the `sample` generator,
        # whose frame would itself hold the client alive) so keep_alive is the
        # ONLY thing keeping the connection alive.
        sampler = client._new_sampler("t", 1, 1, -1)
        del client
        gc.collect()
        sample = sampler.GetNextTrajectory()
        # Tuple layout: (key, probability, table_size, priority, times_sampled,
        # *data columns) — see _BaseClient.sample.
        np.testing.assert_array_equal(np.asarray(sample[5]).reshape(-1), [3.0])
        server.stop()

    def test_structured_writer_outlives_deleted_client(self):
        server, client = _make_shm_server(table_name="sw", max_size=50, min_size=1)
        step_spec = {"a": np.zeros([], np.float32)}
        ref_step = structured_writer.create_reference_step(step_spec)
        pattern = {"x": ref_step["a"][-3:]}
        config = structured_writer.create_config(pattern=pattern, table="sw")
        writer = client.structured_writer(configs=[config])
        del client
        gc.collect()
        for i in range(3):
            writer.append(np.asarray(float(i), dtype=np.float32))
        writer.end_episode()
        client2 = reverb.ShmClient(server.shm_socket_path)
        samples = list(client2.sample("sw", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_array_equal(np.asarray(samples[0].data[0]), [0.0, 1.0, 2.0])
        server.stop()


class ShmClientServerInfoTest(absltest.TestCase):
    """ticket ⑧: server_info() returns real TableInfo over SHM.

    step 1 piggybacked a snapshot on the bootstrap handshake; step 2 added an
    on-demand SERVER_INFO ring round-trip so server_info() reflects live
    table state (current_size after inserts, signature changes) on every call,
    mirroring gRPC's refresh-on-every-call semantics. The round-trip rides the
    INSERT flow under insert_flow_mu like ⑩/⑪'s control-plane ops.
    """

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
        # No inserts yet -> current_size 0 (live round-trip at call time).
        self.assertEqual(info["t"].current_size, 0)

    def test_server_info_reflects_size_at_connect_time(self):
        # Insert items via the in-process path BEFORE the ShmClient connects;
        # the first server_info() call sees them (live round-trip).
        server, _ = _make_shm_server(table_name="t", max_size=10, min_size=1)
        local = server.in_process_client
        for i in range(3):
            _insert_one(local, "t", np.array([float(i)], dtype=np.float32))

        # Now connect an ShmClient; server_info() reflects the 3 items.
        client = reverb.ShmClient(server.shm_socket_path)
        info = client.server_info()
        self.assertEqual(info["t"].current_size, 3)

    def test_server_info_carries_table_signature(self):
        # A table built with a signature propagates it through server_info()
        # so _get_signature_for_table (and thus
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
        # and the signature cache was empty. Now server_info() feeds the cache,
        # so unpacking against the table signature works.
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
        # The hook accepts `timeout` for parity with the gRPC/Local hooks; it is
        # ignored (the SHM round-trip uses its own hard cap, see
        # ShmClient::ServerInfo).
        server, client = _make_shm_server()
        info = client.server_info(timeout=1)
        self.assertIn("t", info)

    def test_server_info_reflects_mid_session_inserts(self):
        # ticket ⑧ step 2: server_info() is a live round-trip, NOT a connect-
        # time snapshot. After connecting on an empty table, insert items via
        # the in-process path and assert a subsequent server_info() call sees
        # the new current_size. Under step 1 (bootstrap snapshot) this would
        # still read 0.
        #
        # Note: we test current_size rather than a Table.replace signature
        # change because Table.replace returns a *new empty* table (the server
        # holds no hot-swap API), so signature mutation mid-session is not
        # reachable from Python. current_size is the honest live-state signal:
        # Table::info() reads data_.size() at call time, so a fresh
        # SERVER_INFO response reflects inserts made since connect.
        server, client = _make_shm_server(table_name="t", max_size=10, min_size=1)
        self.assertEqual(client.server_info()["t"].current_size, 0)
        local = server.in_process_client
        for i in range(4):
            _insert_one(local, "t", np.array([float(i)], dtype=np.float32))
        # The live round-trip sees the 4 items inserted AFTER connect.
        self.assertEqual(client.server_info()["t"].current_size, 4)
        # And again after one more — each call is fresh, not memoized.
        _insert_one(local, "t", np.array([4.0], dtype=np.float32))
        self.assertEqual(client.server_info()["t"].current_size, 5)


class ShmClientValidateItemsTest(absltest.TestCase):
    """ticket ⑧ step 2b: trajectory_writer populates flat_signature_map from
    the live server_info round-trip so CreateItem's ItemAndRefs::Validate runs
    the same signature check as gRPC/LocalClient. Previously the map was left
    empty and validate_items was a no-op."""

    def _make_signed_server(self, sig):
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
        return server, reverb.ShmClient(server.shm_socket_path)

    def test_matching_trajectory_is_accepted(self):
        sig = {"v": signature_codec.TensorSpec((None, 1), np.float32, "v")}
        server, client = self._make_signed_server(sig)
        # Trajectory column matches the signature shape/dtype -> no raise.
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"v": np.array([1.0], dtype=np.float32)})
            w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
            w.flush()

    def test_mismatched_dtype_is_rejected(self):
        sig = {"v": signature_codec.TensorSpec((None, 1), np.float32, "v")}
        server, client = self._make_signed_server(sig)
        # Trajectory column is int but the signature declares float32 ->
        # ValueError (C++ InvalidArgumentError maps to PyExc_ValueError).
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"v": np.array([1], dtype=np.int32)})
            with self.assertRaisesRegex(
                ValueError, "inconsistent with the table signature"
            ):
                w.create_item(
                    table="t",
                    priority=1.0,
                    trajectory={"v": w.history["v"][:]},
                )

    def test_unsigned_table_skips_validation(self):
        # A table with no signature gets nullopt in flat_signature_map, so
        # Validate skips it (any trajectory shape accepted), mirroring
        # LocalClient. _make_shm_server builds an unsigned table.
        server, client = _make_shm_server()
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"v": np.asarray(1.0)})
            w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
            w.flush()


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
        # ticket ⑧-2b: an unknown table name is now rejected at CreateItem
        # time by ItemAndRefs::Validate (the flat_signature_map built from the
        # live server_info round-trip doesn't contain it), surfacing as ValueError
        # (C++ InvalidArgumentError) — mirroring InProcessClient/gRPC
        # validate_items=True. Previously (v1, empty map) it reached the server
        # and surfaced as FileNotFoundError on flush().
        server = self._make_two_table_server()
        client = reverb.ShmClient(server.shm_socket_path)
        with self.assertRaisesRegex(ValueError, "could not be found"):
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
        # ticket ⑧ step 2: server_info() is a live round-trip on both
        # transports, so verify via the in-process client here.
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


class ShmClientCheckpointTest(absltest.TestCase):
    """ticket ⑪: checkpoint() over the SHM transport.

    Mirrors in_process_test.py's InProcessCheckpointTest. checkpoint() rides
    the INSERT flow (insert_c2s/insert_s2c) like ⑩'s control-plane ops; the
    server-side HandleCheckpoint calls the injected checkpointer_->Save over
    ALL tables and returns the path in CheckpointResponse. Restoration is free
    via the shared Table objects (gRPC/InProcess LoadLatest on the same tables
    the SHM server also exposes), so a fresh Server with the same checkpointer
    sees the saved state.
    """

    def _table(self):
        return reverb.Table(
            name="c",
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )

    def test_checkpoint_save_load(self):
        root = tempfile.mkdtemp()
        values = [float(i) for i in range(3)]

        server_a = reverb.Server(
            tables=[self._table()],
            in_process=True,
            shm=True,
            checkpointer=checkpointers.DefaultCheckpointer(path=root),
        )
        client_a = reverb.ShmClient(server_a.shm_socket_path)
        for v in values:
            _insert_one(client_a, "c", np.array([v], dtype=np.float32))

        ckpt_path = client_a.checkpoint()
        self.assertTrue(ckpt_path and os.path.isdir(ckpt_path), ckpt_path)
        for name in ("tables.ckpt", "items.ckpt", "chunks.ckpt", "DONE"):
            self.assertTrue(os.path.exists(os.path.join(ckpt_path, name)), name)

        server_a.stop()

        # Fresh server with the same checkpointer restores the saved state via
        # LoadLatest on the shared Table objects (mirrors InProcessCheckpointTest).
        server_b = reverb.Server(
            tables=[self._table()],
            in_process=True,
            shm=True,
            checkpointer=checkpointers.DefaultCheckpointer(path=root),
        )
        client_b = reverb.ShmClient(server_b.shm_socket_path)
        restored = [
            float(np.asarray(sample.data[0]).reshape(-1)[0])
            for sample in client_b.sample(
                "c", num_samples=len(values), emit_timesteps=False
            )
        ]
        self.assertEqual(restored, values)
        server_b.stop()

    def test_checkpoint_corrupt_raises(self):
        # A CORRUPT checkpoint must surface on Server construction, not be
        # silently swallowed (mirrors InProcessCheckpointTest).
        root = tempfile.mkdtemp()
        server = reverb.Server(
            tables=[self._table()],
            in_process=True,
            shm=True,
            checkpointer=checkpointers.DefaultCheckpointer(path=root),
        )
        client = reverb.ShmClient(server.shm_socket_path)
        ckpt_path = client.checkpoint()
        server.stop()
        with open(os.path.join(ckpt_path, "tables.ckpt"), "wb") as f:
            f.write(b"corrupt-garbage-not-a-proto")
        with self.assertRaises(Exception):
            reverb.Server(
                tables=[self._table()],
                in_process=True,
                shm=True,
                checkpointer=checkpointers.DefaultCheckpointer(path=root),
            )


class ShmSampleDeadlockRegressionTest(absltest.TestCase):
    """Regression for ticket ⑩ 已确认根因：单线程 dispatch 在 HandleSample 的
    rate-limiter 无限阻塞，造成队头阻塞——该 client 的所有后续 ACK 永远排不进
    ring，client ReadBlocking 100% CPU 忙等、timeout 杀不掉。

    修复（方向 A+C）：HandleSample 异步化（EnqueSampleRequest），dispatch 不阻塞；
    client ReadBlocking 加有限超时兑底。本测试复现原卡死场景，断言不再卡死。
    """

    def _make_server(self, min_size=50):
        table = reverb.Table(
            name="t",
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=10000,
            max_times_sampled=1000000,
            rate_limiter=reverb.rate_limiters.MinSize(min_size),
        )
        server = reverb.Server(tables=[table], in_process=True, shm=True)
        return server, reverb.ShmClient(server.shm_socket_path)

    def test_sample_blocking_rate_limiter_does_not_deadlock_mutate(self):
        # min_size=50 但只插 1 条 → sample 在 rate limiter 上等待。修前：dispatch
        # 被这个同步 sample 阻塞，后续 mutate 的 ACK 永远进不了 ring → client
        # ReadBlocking 无限忙等 → timeout 杀不掉。修后：sample 异步入队不卡
        # dispatch，mutate 迅速返回；且即便 dispatch 被卡，ReadBlocking 的 60s
        # 硬上限也保证 client 不无限忙等。
        server, client = self._make_server(min_size=50)
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"v": np.array([1.0], dtype=np.float32)})
            w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
            w.flush()

        result = {}

        def sampler():
            # timeout_ms=None → InfiniteDuration rate-limiter wait。修前这会卡死
            # dispatch；修后 sample 异步入队，dispatch 不阻塞。
            try:
                for s in client.sample("t", num_samples=1, emit_timesteps=False):
                    result["got"] = s
            except Exception as e:  # noqa: BLE001
                result["err"] = repr(e)

        t_s = threading.Thread(target=sampler)
        t_s.start()
        time.sleep(2)  # let the sample request reach the server

        def mutate():
            try:
                client.mutate_priorities("t", updates={0: 1.0})
                result["mutate_ok"] = True
            except Exception as e:  # noqa: BLE001
                result["mutate_err"] = repr(e)

        t_m = threading.Thread(target=mutate)
        t_m.start()
        # mutate must return promptly (well under the 60s ReadBlocking cap).
        # 修前会无限阻塞。给 10s 上限（远小于 60s 兑底，证明是 A 的根治而非 C 的超时）。
        t_m.join(timeout=10)
        self.assertFalse(
            t_m.is_alive(),
            "mutate thread hung — dispatch head-of-line block not fixed",
        )
        # mutate succeeded (key 0 may be absent; MutateItems ignores it).
        self.assertNotIn("mutate_err", result, f"mutate errored: {result}")
        self.assertTrue(result.get("mutate_ok"), f"mutate did not complete: {result}")

        # The sampler is still blocked on the rate limiter (min_size=50 never
        # reached). Kill it cleanly via server.stop (releases the table) — the
        # sampler should surface an error/cancel, not hang.
        server.stop()
        t_s.join(timeout=15)
        self.assertFalse(t_s.is_alive(), "sampler thread hung after server stop")


if __name__ == "__main__":
    absltest.main()
