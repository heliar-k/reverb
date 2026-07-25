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

"""End-to-end test for the in-process / numpy-only Reverb mode.

Exercises the full write -> create_item -> flush -> sample round trip using
plain numpy arrays, without TensorFlow. Migrated to absltest from the
historical bare-assert + __main__ form.
"""

import os
import tempfile
import threading
import time

import numpy as np
from absl.testing import absltest

import reverb
from reverb import errors, replay_sample, signature_codec, structured_writer
from reverb.platform.default import checkpointers


def _make_server(
    table_name="t",
    max_size=10,
    min_size=1,
    sampler=None,
    remover=None,
    max_times_sampled=1,
):
    """Builds a single-table in-process server with the given strategies.

    `max_times_sampled` defaults to 1 to mirror `Table.queue`: an item is removed
    after one draw, so sampling N items returns N distinct items. Tests that need
    over-sampling (drawing more items than were inserted) must pass
    `max_times_sampled=0` explicitly.
    """
    return reverb.Server(
        tables=[
            reverb.Table(
                name=table_name,
                sampler=sampler or reverb.selectors.Fifo(),
                remover=remover or reverb.selectors.Fifo(),
                max_size=max_size,
                max_times_sampled=max_times_sampled,
                rate_limiter=reverb.rate_limiters.MinSize(min_size),
            )
        ],
        in_process=True,
    )


def _insert_one(client, table, value, priority=1.0, num_keep_alive_refs=1):
    """Inserts a single-item trajectory carrying one scalar-bearing column."""
    with client.trajectory_writer(num_keep_alive_refs=num_keep_alive_refs) as w:
        w.append({"v": np.asarray(value)})
        w.create_item(
            table=table, priority=priority, trajectory={"v": w.history["v"][:]}
        )
        w.flush()


class InProcessWriteSampleTest(absltest.TestCase):
    def test_in_process_write_sample(self):
        server = _make_server()
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": np.array([1.0, 2.0], dtype=np.float32)})
            w.create_item(
                table="t", priority=1.0, trajectory={"obs": w.history["obs"][:]}
            )
            w.flush()

        samples = list(client.sample("t", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)
        np.testing.assert_allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]])

    def test_in_process_multiple_steps(self):
        server = _make_server(table_name="q")
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=3) as w:
            for i in range(3):
                w.append({"obs": np.array([float(i)], dtype=np.float32)})
            w.create_item(
                table="q",
                priority=1.0,
                trajectory={"obs": w.history["obs"][:]},
            )
            w.flush()

        samples = list(client.sample("q", num_samples=1, emit_timesteps=False))
        np.testing.assert_allclose(
            np.asarray(samples[0].data[0]), [[0.0], [1.0], [2.0]]
        )

    def test_in_process_mutate_and_reset(self):
        server = _make_server()
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": np.array([42.0], dtype=np.float32)})
            w.create_item(
                table="t", priority=1.0, trajectory={"obs": w.history["obs"][:]}
            )
            w.flush()

        info = client.server_info()
        self.assertIn("t", info)
        self.assertEqual(info["t"].max_size, 10)

        client.reset("t")
        info_after = client.server_info()
        self.assertEqual(info_after["t"].current_size, 0)


class InProcessSelectorTest(absltest.TestCase):
    def test_uniform_replay(self):
        # Over-samples (5 inserted, 20 drawn) so items must be reusable.
        server = _make_server(
            table_name="u",
            max_size=50,
            min_size=1,
            sampler=reverb.selectors.Uniform(),
            max_times_sampled=0,
        )
        client = server.in_process_client

        values = [float(i) for i in range(5)]
        for v in values:
            _insert_one(client, "u", np.array([v], dtype=np.float32))

        seen = set()
        for sample in client.sample("u", num_samples=20, emit_timesteps=False):
            seen.add(float(np.asarray(sample.data[0]).reshape(-1)[0]))
        self.assertTrue(seen.issubset(set(values)), seen)
        self.assertGreaterEqual(len(seen), 3, seen)

    def test_lifo_replay(self):
        server = _make_server(
            table_name="l",
            max_size=10,
            min_size=1,
            sampler=reverb.selectors.Lifo(),
            max_times_sampled=1,
        )
        client = server.in_process_client

        for i in range(4):
            _insert_one(client, "l", np.array([float(i)], dtype=np.float32))

        order = [
            float(np.asarray(sample.data[0]).reshape(-1)[0])
            for sample in client.sample("l", num_samples=4, emit_timesteps=False)
        ]
        self.assertEqual(order, [3.0, 2.0, 1.0, 0.0])

    def test_prioritized_replay(self):
        # Over-samples (2 inserted, 100 drawn) so items must be reusable.
        server = _make_server(
            table_name="p",
            max_size=20,
            min_size=1,
            sampler=reverb.selectors.Prioritized(1.0),
            max_times_sampled=0,
        )
        client = server.in_process_client

        _insert_one(client, "p", np.array([0.0], dtype=np.float32), priority=0.1)
        _insert_one(client, "p", np.array([1.0], dtype=np.float32), priority=100.0)

        counts = {0.0: 0, 1.0: 0}
        for sample in client.sample("p", num_samples=100, emit_timesteps=False):
            val = float(np.asarray(sample.data[0]).reshape(-1)[0])
            counts[val] += 1
        self.assertGreater(counts[1.0], counts[0.0], counts)


class InProcessDtypesTest(absltest.TestCase):
    def test_multiple_dtypes(self):
        server = _make_server(
            table_name="d", max_size=10, min_size=1, max_times_sampled=1
        )
        client = server.in_process_client

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


class InProcessBytesDtypeTest(absltest.TestCase):
    """Regression: numpy bytes (S dtype) must round-trip byte-for-byte.

    tensor_proxy.cc FromNdArray serialized string arrays via py::str(item),
    turning b'abc' into the 6-char repr \"b'abc'\" — silent data corruption.
    """

    def test_bytes_array_round_trip(self):
        server = _make_server(table_name="b", max_size=10, min_size=1)
        client = server.in_process_client

        arr = np.array([b"abc", b"de"])  # dtype '|S3'
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"v": arr})
            w.create_item(
                table="b", priority=1.0, trajectory={"v": w.history["v"][:]}
            )
            w.flush()

        sample = next(client.sample("b", num_samples=1, emit_timesteps=False))
        got = np.asarray(sample.data[0])
        self.assertEqual(got.dtype, arr.dtype, (got.dtype, arr.dtype))
        np.testing.assert_array_equal(got, [arr])


class InProcessRateLimiterTest(absltest.TestCase):
    def test_rate_limiter_min_size(self):
        # MinSize(5) blocks sampling while the table holds < 5 items. With
        # max_times_sampled=1 each draw deletes an item, so drawing 5 would drain
        # the table below the limit mid-way and deadlock; keep items reusable.
        server = _make_server(
            table_name="r",
            max_size=20,
            min_size=5,
            sampler=reverb.selectors.Fifo(),
            max_times_sampled=0,
        )
        client = server.in_process_client

        for i in range(5):
            _insert_one(client, "r", np.array([float(i)], dtype=np.float32))

        samples = list(client.sample("r", num_samples=5, emit_timesteps=False))
        self.assertLen(samples, 5)

    def test_rate_limiter_timeout(self):
        server = _make_server(
            table_name="r", max_size=20, min_size=5, sampler=reverb.selectors.Fifo()
        )
        client = server.in_process_client

        for i in range(2):
            _insert_one(client, "r", np.array([float(i)], dtype=np.float32))

        start = time.time()
        with self.assertRaises(errors.DeadlineExceededError):
            list(
                client.sample("r", num_samples=1, timeout_ms=500, emit_timesteps=False)
            )
        elapsed = time.time() - start
        self.assertLess(elapsed, 5.0, f"timeout took too long: {elapsed:.1f}s")

    def test_rate_limiter_timeout_none_compatible(self):
        server = _make_server(
            table_name="n", max_size=10, min_size=1, sampler=reverb.selectors.Fifo()
        )
        client = server.in_process_client
        _insert_one(client, "n", np.array([42.0], dtype=np.float32))
        samples = list(client.sample("n", num_samples=1, emit_timesteps=False))
        self.assertLen(samples, 1)

    def test_raw_sampler_raises_deadline_exceeded(self):
        # MinSize(5), no items: the raw pybind Sampler.GetNextTrajectory must raise
        # reverb.errors.DeadlineExceededError directly (not RuntimeError).
        srv = reverb.Server(
            tables=[
                reverb.Table(
                    name="blocked",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=10,
                    rate_limiter=reverb.rate_limiters.MinSize(5),
                )
            ],
            in_process=True,
        )
        c = srv.in_process_client
        sampler = c.new_sampler("blocked", num_samples=1, timeout_ms=200)
        with self.assertRaises(errors.DeadlineExceededError):
            sampler.GetNextTrajectory()


class InProcessCheckpointTest(absltest.TestCase):
    def test_checkpoint_save_load(self):
        root = tempfile.mkdtemp()
        values = [float(i) for i in range(3)]

        table = lambda: reverb.Table(
            name="c",
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )
        server_a = reverb.Server(
            tables=[table()],
            in_process=True,
            checkpointer=checkpointers.DefaultCheckpointer(path=root),
        )
        client_a = server_a.in_process_client
        for v in values:
            _insert_one(client_a, "c", np.array([v], dtype=np.float32))

        ckpt_path = client_a.checkpoint()
        self.assertTrue(ckpt_path and os.path.isdir(ckpt_path), ckpt_path)
        for name in ("tables.ckpt", "items.ckpt", "chunks.ckpt", "DONE"):
            self.assertTrue(os.path.exists(os.path.join(ckpt_path, name)), name)

        del server_a

        server_b = reverb.Server(
            tables=[table()],
            in_process=True,
            checkpointer=checkpointers.DefaultCheckpointer(path=root),
        )
        client_b = server_b.in_process_client

        restored = [
            float(np.asarray(sample.data[0]).reshape(-1)[0])
            for sample in client_b.sample(
                "c", num_samples=len(values), emit_timesteps=False
            )
        ]
        self.assertEqual(restored, values)

    def test_corrupt_checkpoint_raises(self):
        # A CORRUPT checkpoint must surface as an exception on Server construction,
        # not be silently swallowed (which would mean silent data loss).
        root = tempfile.mkdtemp()
        table = lambda: reverb.Table(
            name="c",
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )
        srv = reverb.Server(
            tables=[table()],
            in_process=True,
            checkpointer=checkpointers.DefaultCheckpointer(path=root),
        )
        ckpt_path = srv.in_process_client.checkpoint()
        srv.stop()
        # Corrupt the checkpoint: overwrite tables.ckpt with garbage.
        with open(os.path.join(ckpt_path, "tables.ckpt"), "wb") as f:
            f.write(b"corrupt-garbage-not-a-proto")
        # Constructing a new server must NOT silently swallow the corruption.
        with self.assertRaises(Exception):
            reverb.Server(
                tables=[table()],
                in_process=True,
                checkpointer=checkpointers.DefaultCheckpointer(path=root),
            )

    def test_first_start_no_checkpoint_is_benign(self):
        # A FRESH empty checkpointer dir must NOT raise on Server construction.
        root = tempfile.mkdtemp()  # empty
        table = lambda: reverb.Table(
            name="c",
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )
        srv = reverb.Server(
            tables=[table()],
            in_process=True,
            checkpointer=checkpointers.DefaultCheckpointer(path=root),
        )
        # And the table is empty (no phantom data restored).
        self.assertEqual(srv.in_process_client.server_info()["c"].current_size, 0)


class InProcessSignatureUnpackTest(absltest.TestCase):
    """LocalClient.sample unpack_as_table_signature support."""

    def test_unpack_returns_structured_data(self):
        sig = {
            "obs": signature_codec.TensorSpec((None, 1), np.float32, "obs"),
            "action": signature_codec.TensorSpec((None,), np.int32, "action"),
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
        )
        client = server.in_process_client

        # 写入一个 3 步轨迹
        with client.trajectory_writer(num_keep_alive_refs=3) as w:
            for i in range(3):
                w.append(
                    {
                        "obs": np.array([float(i)], dtype=np.float32),
                        "action": np.array(i, dtype=np.int32),
                    }
                )
            w.create_item(
                table="t",
                priority=1.0,
                trajectory={
                    "obs": w.history["obs"][:],
                    "action": w.history["action"][:],
                },
            )
            w.flush()

        # unpack_as_table_signature=True -> data 是 dict
        sample = next(
            client.sample(
                "t", num_samples=1, unpack_as_table_signature=True, emit_timesteps=False
            )
        )
        self.assertIsInstance(sample.data, dict)
        self.assertEqual(set(sample.data.keys()), {"obs", "action"})
        np.testing.assert_array_equal(sample.data["obs"], [[0.0], [1.0], [2.0]])
        np.testing.assert_array_equal(sample.data["action"], [0, 1, 2])

    def test_unpack_false_returns_flat(self):
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
        )
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"obs": np.array([42.0], dtype=np.float32)})
            w.create_item(
                table="t", priority=1.0, trajectory={"obs": w.history["obs"][:]}
            )
            w.flush()

        # unpack_as_table_signature=False -> data 是 flat list
        sample = next(
            client.sample(
                "t",
                num_samples=1,
                unpack_as_table_signature=False,
                emit_timesteps=False,
            )
        )
        self.assertIsInstance(sample.data, list)
        np.testing.assert_array_equal(np.asarray(sample.data[0]), [[42.0]])

    def test_unpack_table_without_signature_returns_flat(self):
        # 表没声明 signature -> 即使 unpack=True 也是 flat
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
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"v": np.array([1.0], dtype=np.float32)})
            w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
            w.flush()

        sample = next(
            client.sample(
                "t", num_samples=1, unpack_as_table_signature=True, emit_timesteps=False
            )
        )
        # signature=None -> flat list
        self.assertIsInstance(sample.data, list)

    def test_unpack_unknown_table_raises(self):
        server = reverb.Server(
            tables=[
                reverb.Table(
                    name="t",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=10,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                )
            ],
            in_process=True,
        )
        client = server.in_process_client
        with self.assertRaises(ValueError):
            next(
                client.sample(
                    "nonexistent",
                    num_samples=1,
                    unpack_as_table_signature=True,
                    timeout_ms=100,
                )
            )


class InProcessStructuredWriterTest(absltest.TestCase):
    """LocalClient.structured_writer end-to-end (in-process / numpy).

    Mirrors the gRPC `StructuredWriterTest` flow but routes through the
    in-process `LocalClient`, whose writer is bound to a single table.
    """

    def test_trajectory_pattern(self):
        # Mirror of gRPC test_trajectory_patterns: a 3-window over a scalar
        # column yields one trajectory of length 3.
        server = _make_server(table_name="sw", max_size=50, min_size=1)
        client = server.in_process_client

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

    def test_empty_configs_raises(self):
        server = _make_server(table_name="sw", max_size=10, min_size=1)
        client = server.in_process_client
        with self.assertRaises(ValueError):
            client.structured_writer(configs=[])

    def test_single_condition_inserts_matching_steps(self):
        # Mirror of gRPC test_single_condition with Condition.step_index() <= 2
        # and a relative scalar slice x[-1]. Each matching step must yield a
        # one-element trajectory carrying that step's value.
        #
        # max_times_sampled=1 mirrors the gRPC Table.queue default; without it
        # (max_times_sampled=0) the Fifo sampler re-returns the oldest item on
        # every draw, yielding [0,0,0] by correct sampler behaviour rather than an
        # engine bug. See structured_writer.cc / in_process_client.cc: the
        # StructuredWriter inserts three distinct items (values 0,1,2) on both
        # paths.
        server = _make_server(
            table_name="sw", max_size=50, min_size=1, max_times_sampled=1
        )
        client = server.in_process_client

        pattern = structured_writer.pattern_from_transform(
            step_structure=None, transform=lambda x: x[-1]
        )
        config = structured_writer.create_config(
            pattern=pattern,
            table="sw",
            conditions=[structured_writer.Condition.step_index() <= 2],
        )
        writer = client.structured_writer(configs=[config])
        for i in range(5):
            writer.append(i)
        writer.end_episode()

        samples = list(client.sample("sw", num_samples=3, emit_timesteps=False))
        got = [int(np.asarray(s.data[0]).reshape(-1)[0]) for s in samples]
        self.assertEqual(got, [0, 1, 2])


class InProcessSignatureCacheRefreshTest(absltest.TestCase):
    """Regression for A4: server_info() must refresh the signature cache.

    Previously the cache was only populated when empty ("if this is the first
    time"), so once filled it never reflected table replacements. Now every
    server_info() call re-derives the cache, mirroring the C++
    Client::ServerInfo "Forces an update of internal signature caches" semantics.
    """

    def _server_with_signature(self, sig):
        return reverb.Server(
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
        )

    def test_server_info_refreshes_cache_on_every_call(self):
        # The cache must track the live server state, not freeze after the first
        # successful call. We corrupt the cache after the first call and assert
        # that a second call restores it to the real signature.
        sig = {"v": signature_codec.TensorSpec((None, 1), np.float32, "v")}
        client = self._server_with_signature(sig).in_process_client

        client.server_info()
        self.assertEqual(set(client._signature_cache.keys()), {"t"})
        cached_before = client._signature_cache["t"]

        # Simulate a stale/frozen cache (the bug: non-empty cache was never
        # overwritten). Clobber it with a bogus value.
        client._signature_cache = {"t": "STALE_BOGUS"}

        info2 = client.server_info()
        # Cache must now reflect the refreshed value, not the clobbered one.
        self.assertNotEqual(client._signature_cache["t"], "STALE_BOGUS")
        self.assertEqual(client._signature_cache["t"], cached_before)
        self.assertEqual(client._signature_cache["t"], info2["t"].signature)

    def test_cache_matches_returned_signature_after_refresh(self):
        # The refreshed cache must equal the signature carried by the very
        # TableInfo that server_info() returns.
        sig = {"v": signature_codec.TensorSpec((None,), np.int32, "v")}
        client = self._server_with_signature(sig).in_process_client
        client.server_info()  # populate
        # Second call on a now-populated cache must still refresh and stay
        # consistent with the returned TableInfo.
        info = client.server_info()
        self.assertEqual(client._signature_cache["t"], info["t"].signature)


def _make_grpc_table(table_name="t", max_size=10, min_size=1, max_times_sampled=1):
    return reverb.Table(
        name=table_name,
        sampler=reverb.selectors.Fifo(),
        remover=reverb.selectors.Fifo(),
        max_size=max_size,
        max_times_sampled=max_times_sampled,
        rate_limiter=reverb.rate_limiters.MinSize(min_size),
    )


class ClientLocalClientParityTest(absltest.TestCase):
    """Regression guard for the _BaseClient convergence (task A3).

    Asserts that the gRPC `Client` and the in-process `LocalClient` agree on the
    shared `_BaseClient` surface: `server_info`, `sample`, `mutate_priorities`,
    `reset`, and `checkpoint`. Each test spins up a fresh (gRPC, local) server
    pair with the sampler semantics it needs.
    """

    TABLE = "t"

    def _make_pair(self, max_times_sampled=1):
        """Returns (grpc_client, grpc_server, local_client, local_server)."""
        grpc_server = reverb.Server(
            tables=[_make_grpc_table(self.TABLE, max_times_sampled=max_times_sampled)]
        )
        grpc = grpc_server.localhost_client()
        local_server = _make_server(max_times_sampled=max_times_sampled)
        return grpc, grpc_server, local_server.in_process_client, local_server

    def _insert_both(self, grpc, local, value):
        # Both clients now share `_BaseClient.insert` (single implementation,
        # dispatched via `self.writer`/`self._client.NewWriter`). This is the
        # real API-alignment regression: previously LocalClient had no `insert`
        # and was mirrored via the trajectory writer.
        grpc.insert(np.asarray(value), {self.TABLE: 1.0})
        local.insert(np.asarray(value), {self.TABLE: 1.0})

    def test_server_info_parity(self):
        grpc, grpc_server, local, local_server = self._make_pair()
        try:
            self._insert_both(grpc, local, 1)
            grpc_info = grpc.server_info()
            local_info = local.server_info()
            self.assertEqual(set(grpc_info), {self.TABLE})
            self.assertEqual(set(local_info), {self.TABLE})
            self.assertEqual(
                grpc_info[self.TABLE].max_size, local_info[self.TABLE].max_size
            )
            self.assertEqual(
                grpc_info[self.TABLE].current_size, local_info[self.TABLE].current_size
            )
        finally:
            grpc_server.stop()

    def _val(self, sample):
        return float(np.asarray(sample.data[0]).reshape(-1)[0])

    def test_sample_parity_fifo(self):
        # max_times_sampled=1 => each item sampled once; FIFO returns them in
        # insertion order, deterministically, on both clients.
        grpc, grpc_server, local, local_server = self._make_pair(max_times_sampled=1)
        try:
            for v in (10, 20, 30):
                self._insert_both(grpc, local, v)

            grpc_samples = list(
                grpc.sample(self.TABLE, num_samples=3, emit_timesteps=False)
            )
            local_samples = list(
                local.sample(self.TABLE, num_samples=3, emit_timesteps=False)
            )
            self.assertLen(grpc_samples, 3)
            self.assertLen(local_samples, 3)
            self.assertEqual(
                [self._val(s) for s in grpc_samples],
                [self._val(s) for s in local_samples],
            )
            self.assertEqual([self._val(s) for s in grpc_samples], [10.0, 20.0, 30.0])
        finally:
            grpc_server.stop()

    def test_mutate_priorities_parity(self):
        # Single item + max_times_sampled=0 (unlimited) so the same key can be
        # sampled before and after the priority mutation on both clients.
        grpc, grpc_server, local, local_server = self._make_pair(max_times_sampled=0)
        try:
            self._insert_both(grpc, local, 1)

            grpc_key = next(grpc.sample(self.TABLE, 1, emit_timesteps=False)).info.key
            local_key = next(local.sample(self.TABLE, 1, emit_timesteps=False)).info.key

            # Mutating priorities must not raise on either client and must surface
            # the new priority on the next sample of the same key.
            grpc.mutate_priorities(self.TABLE, updates={grpc_key: 42.0})
            local.mutate_priorities(self.TABLE, updates={local_key: 42.0})
            grpc_prio = next(
                grpc.sample(self.TABLE, 1, emit_timesteps=False)
            ).info.priority
            local_prio = next(
                local.sample(self.TABLE, 1, emit_timesteps=False)
            ).info.priority
            self.assertEqual(grpc_prio, 42.0)
            self.assertEqual(local_prio, 42.0)
        finally:
            grpc_server.stop()

    def test_reset_parity(self):
        grpc, grpc_server, local, local_server = self._make_pair()
        try:
            self._insert_both(grpc, local, 1)
            self._insert_both(grpc, local, 2)
            grpc.reset(self.TABLE)
            local.reset(self.TABLE)
            self.assertEqual(grpc.server_info()[self.TABLE].current_size, 0)
            self.assertEqual(local.server_info()[self.TABLE].current_size, 0)
        finally:
            grpc_server.stop()

    def test_checkpoint_parity(self):
        grpc, grpc_server, local, local_server = self._make_pair()
        try:
            self._insert_both(grpc, local, 1)
            grpc_path = grpc.checkpoint()
            local_path = local.checkpoint()
            self.assertIsInstance(grpc_path, str)
            self.assertIsInstance(local_path, str)
            self.assertTrue(grpc_path)
            self.assertTrue(local_path)
        finally:
            grpc_server.stop()

    def test_sample_emit_timesteps_defaults_match(self):
        # After unchaining, both gRPC `Client` and in-process `LocalClient`
        # default `sample(emit_timesteps=...)` to True (the historical gRPC
        # behavior). The previous LocalClient=False default was a side effect of
        # the single-table binding; it is gone now. Callers can still override.
        grpc, grpc_server, local, local_server = self._make_pair(max_times_sampled=0)
        try:
            self._insert_both(grpc, local, 1)

            # Both defaults: True -> a list of per-timestep ReplaySample.
            local_default = next(local.sample(self.TABLE, 1))
            grpc_default = next(grpc.sample(self.TABLE, 1))
            self.assertIsInstance(local_default, list)
            self.assertIsInstance(grpc_default, list)
            self.assertTrue(
                all(isinstance(ts, replay_sample.ReplaySample) for ts in local_default)
            )
            self.assertTrue(
                all(isinstance(ts, replay_sample.ReplaySample) for ts in grpc_default)
            )

            # Both honor an explicit override to False (whole trajectory).
            local_as_whole = next(local.sample(self.TABLE, 1, emit_timesteps=False))
            grpc_as_whole = next(grpc.sample(self.TABLE, 1, emit_timesteps=False))
            self.assertIsInstance(local_as_whole, replay_sample.ReplaySample)
            self.assertIsInstance(grpc_as_whole, replay_sample.ReplaySample)
        finally:
            grpc_server.stop()

    def test_writer_parity(self):
        # D3-a regression: the new local Writer path must produce data
        # indistinguishable from the gRPC Writer when fed the same steps.
        grpc, grpc_server, local, local_server = self._make_pair(max_times_sampled=1)
        try:
            with grpc.writer(max_sequence_length=1) as w:
                w.append(np.asarray(42.0))
                w.create_item(self.TABLE, num_timesteps=1, priority=1.0)
            with local.writer(max_sequence_length=1) as w:
                w.append(np.asarray(42.0))
                w.create_item(self.TABLE, num_timesteps=1, priority=1.0)

            grpc_sample = next(grpc.sample(self.TABLE, 1, emit_timesteps=False))
            local_sample = next(local.sample(self.TABLE, 1, emit_timesteps=False))
            self.assertEqual(self._val(grpc_sample), self._val(local_sample))
            self.assertEqual(self._val(grpc_sample), 42.0)
        finally:
            grpc_server.stop()

    def test_insert_parity(self):
        # `_insert_both` now uses `insert` on both clients (real API parity).
        # Explicitly assert the sampled values match across backends.
        grpc, grpc_server, local, local_server = self._make_pair(max_times_sampled=1)
        try:
            self._insert_both(grpc, local, 7)
            grpc_sample = next(grpc.sample(self.TABLE, 1, emit_timesteps=False))
            local_sample = next(local.sample(self.TABLE, 1, emit_timesteps=False))
            self.assertEqual(self._val(grpc_sample), self._val(local_sample))
            self.assertEqual(self._val(grpc_sample), 7.0)
        finally:
            grpc_server.stop()


class LocalClientInsertWriterTest(absltest.TestCase):
    """Availability tests for LocalClient.insert / writer (D3-a).

    These confirm the local paths exist and behave like their gRPC
    counterparts when used standalone (not via the parity pair).
    """

    def test_local_insert_is_available(self):
        server = _make_server(table_name="t", max_size=10, min_size=1)
        client = server.in_process_client
        client.insert(np.asarray(5.0), {"t": 1.0})
        sample = next(client.sample("t", 1, emit_timesteps=False))
        got = float(np.asarray(sample.data[0]).reshape(-1)[0])  # pylint: disable=bare-except
        self.assertEqual(got, 5.0)

    def test_local_writer_is_available(self):
        server = _make_server(table_name="t", max_size=10, min_size=1)
        client = server.in_process_client
        with client.writer(max_sequence_length=1) as w:
            w.append(np.asarray(9.0))
            w.create_item("t", num_timesteps=1, priority=1.0)
        sample = next(client.sample("t", 1, emit_timesteps=False))
        got = float(np.asarray(sample.data[0]).reshape(-1)[0])  # pylint: disable=bare-except
        self.assertEqual(got, 9.0)

    def test_local_insert_multiple_tables(self):
        # Multi-table insert must work locally: a single `insert` can target
        # more than one table (the unchained writer routes by item.table()).
        server = reverb.Server(
            tables=[
                reverb.Table(
                    name="a",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=10,
                    max_times_sampled=1,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                ),
                reverb.Table(
                    name="b",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=10,
                    max_times_sampled=1,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                ),
            ],
            in_process=True,
        )
        client = server.in_process_client
        client.insert(np.asarray(3.0), {"a": 1.0, "b": 2.0})
        self.assertEqual(client.server_info()["a"].current_size, 1)
        self.assertEqual(client.server_info()["b"].current_size, 1)


class TrajectoryWriterConfigureTest(absltest.TestCase):
    """configure() against a real in-process writer (not a mock).

    Mirrors trajectory_writer_test.test_configure_seen_column but routes through
    LocalClient.trajectory_writer so the C++ ConfigureChunker path is actually
    exercised end-to-end.
    """

    def test_configure_seen_column_round_trip(self):
        server = _make_server(table_name="t", max_size=10, min_size=1)
        client = server.in_process_client

        w = client.trajectory_writer(num_keep_alive_refs=3, max_chunk_length=2)
        # Establish the 'v' column.
        w.append({"v": np.array([1.0], dtype=np.float32)})
        # The C++ chunker refuses ApplyConfig while its buffer holds unflushed
        # data, so finalize the column into a chunk first via an item + flush.
        w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
        w.flush()
        # Reconfigure the seen column to a constant chunk length of 1.
        w.configure(("v",), num_keep_alive_refs=3, max_chunk_length=1)

        # The writer must remain usable after configure.
        w.append({"v": np.array([2.0], dtype=np.float32)})
        w.append({"v": np.array([3.0], dtype=np.float32)})
        w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
        w.end_episode()

        # Two items inserted (the flushed 1-step one + the 3-step one); FIFO
        # returns the 3-step one second.
        samples = list(client.sample("t", num_samples=2, emit_timesteps=False))
        got = [np.asarray(s.data[0]).tolist() for s in samples]
        self.assertIn([[1.0], [2.0], [3.0]], got)


class TrajectoryWriterFlushTimeoutTest(absltest.TestCase):
    """in-process flush(timeout_ms=...) surfaces DeadlineExceededError.

    Mirrors trajectory_writer_test.test_timeout_on_flush but via LocalClient.
    A Queue(1) limiter blocks the second insert (0 samples drawn), so items pile
    up unconfirmed in the writer's in-flight set and flush times out.
    """

    def _blocked_server(self):
        return reverb.Server(
            tables=[
                reverb.Table(
                    name="t",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=1,
                    max_times_sampled=1,
                    rate_limiter=reverb.rate_limiters.Queue(1),
                )
            ],
            in_process=True,
        )

    def test_flush_times_out(self):
        client = self._blocked_server().in_process_client
        w = client.trajectory_writer(num_keep_alive_refs=1)
        w.append({"v": np.array([1.0], dtype=np.float32)})

        with self.assertRaises(errors.DeadlineExceededError):
            for _ in range(4):
                w.create_item(
                    table="t", priority=1.0, trajectory={"v": w.history["v"][:]}
                )
                w.flush(timeout_ms=1)


class TrajectoryWriterEndEpisodeTimeoutTest(absltest.TestCase):
    """in-process end_episode(timeout_ms=...) surfaces DeadlineExceededError."""

    def _blocked_server(self):
        return reverb.Server(
            tables=[
                reverb.Table(
                    name="t",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=1,
                    max_times_sampled=1,
                    rate_limiter=reverb.rate_limiters.Queue(1),
                )
            ],
            in_process=True,
        )

    def test_end_episode_times_out(self):
        client = self._blocked_server().in_process_client
        w = client.trajectory_writer(num_keep_alive_refs=1)
        w.append({"v": np.array([1.0], dtype=np.float32)})

        with self.assertRaises(errors.DeadlineExceededError):
            for _ in range(4):
                w.create_item(
                    table="t", priority=1.0, trajectory={"v": w.history["v"][:]}
                )
                w.end_episode(clear_buffers=False, timeout_ms=1)


class TrajectoryWriterEndEpisodeClearBuffersFalseTest(absltest.TestCase):
    """end_episode(clear_buffers=False) keeps buffers across episodes."""

    def test_history_persists_across_episodes(self):
        server = _make_server(table_name="t", max_size=10, min_size=1)
        client = server.in_process_client
        w = client.trajectory_writer(num_keep_alive_refs=3)

        # Episode 1: two steps.
        w.append({"v": np.array([1.0], dtype=np.float32)})
        w.append({"v": np.array([2.0], dtype=np.float32)})
        w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
        w.end_episode(clear_buffers=False)

        # episode_steps reset, but the 'v' buffer must still hold episode 1.
        self.assertEqual(w.episode_steps, 0)
        self.assertEqual(len(w.history["v"]), 2)

        # Episode 2: one more step; the new item spans both episodes.
        w.append({"v": np.array([3.0], dtype=np.float32)})
        self.assertEqual(len(w.history["v"]), 3)
        w.create_item(table="t", priority=1.0, trajectory={"v": w.history["v"][:]})
        w.end_episode()

        # Two items inserted (episode-1's 2-step and the cross-episode 3-step);
        # the 3-step one carries data from both episodes.
        samples = list(client.sample("t", num_samples=2, emit_timesteps=False))
        got = [np.asarray(s.data[0]).tolist() for s in samples]
        self.assertIn([[1.0], [2.0], [3.0]], got)


class TrajectoryWriterAppendDtypeMismatchTest(absltest.TestCase):
    """append dtype mismatch raises ValueError naming the structured path."""

    def test_dtype_mismatch_names_column(self):
        server = _make_server(table_name="t", max_size=10, min_size=1)
        client = server.in_process_client
        w = client.trajectory_writer(num_keep_alive_refs=1)

        w.append({"v": np.array([1.0], dtype=np.float32)})
        with self.assertRaises(ValueError) as ctx:
            w.append({"v": np.array([2], dtype=np.int32)})
        # The C++ "for column N" message must be rewritten to the path 'v'.
        self.assertIn("'v'", str(ctx.exception))


class InProcessWaitTest(absltest.TestCase):
    """Server(in_process=True).wait() must block until stop().

    Regression: wait() only called the gRPC server's Wait(), so in-process
    (and SHM-only) servers returned instantly despite the docstring promising
    to block until shutdown (server.py:506).
    """

    def test_wait_blocks_until_stop(self):
        server = _make_server()
        t = threading.Thread(target=server.wait)
        t.start()
        time.sleep(0.3)
        self.assertTrue(t.is_alive(), "wait() returned before stop()")
        server.stop()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "wait() did not return after stop()")


class MultiTableServerInfoTest(absltest.TestCase):
    """server_info reflects every table on a multi-table LocalClient."""

    def test_two_tables_reported(self):
        server = reverb.Server(
            tables=[
                reverb.Table(
                    name="t1",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=7,
                    max_times_sampled=1,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                ),
                reverb.Table(
                    name="t2",
                    sampler=reverb.selectors.Fifo(),
                    remover=reverb.selectors.Fifo(),
                    max_size=3,
                    max_times_sampled=1,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                ),
            ],
            in_process=True,
        )
        client = server.in_process_client

        info = client.server_info()
        self.assertEqual(set(info), {"t1", "t2"})
        self.assertEqual(info["t1"].max_size, 7)
        self.assertEqual(info["t2"].max_size, 3)
        self.assertEqual(info["t1"].current_size, 0)
        self.assertEqual(info["t2"].current_size, 0)


if __name__ == "__main__":
    absltest.main()
