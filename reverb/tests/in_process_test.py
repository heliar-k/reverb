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
import time

from absl.testing import absltest
import numpy as np

import reverb
from reverb import errors
from reverb import signature_codec
from reverb.platform.default import checkpointers


def _make_server(table_name='t', max_size=10, min_size=1,
                 sampler=None, remover=None, max_times_sampled=0):
  """Builds a single-table in-process server with the given strategies."""
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
  with client.trajectory_writer(table=table,
                                num_keep_alive_refs=num_keep_alive_refs) as w:
    w.append({'v': np.asarray(value)})
    w.create_item(
        table=table, priority=priority,
        trajectory={'v': w.history['v'][:]})
    w.flush()


class InProcessWriteSampleTest(absltest.TestCase):

  def test_in_process_write_sample(self):
    server = _make_server()
    client = server.in_process_client

    with client.trajectory_writer(table='t', num_keep_alive_refs=1) as w:
      w.append({'obs': np.array([1.0, 2.0], dtype=np.float32)})
      w.create_item(
          table='t', priority=1.0, trajectory={'obs': w.history['obs'][:]})
      w.flush()

    samples = list(client.sample('t', num_samples=1, emit_timesteps=False))
    self.assertLen(samples, 1)
    np.testing.assert_allclose(
        np.asarray(samples[0].data[0]), [[1.0, 2.0]])

  def test_in_process_multiple_steps(self):
    server = _make_server(table_name='q')
    client = server.in_process_client

    with client.trajectory_writer(table='q', num_keep_alive_refs=3) as w:
      for i in range(3):
        w.append({'obs': np.array([float(i)], dtype=np.float32)})
      w.create_item(
          table='q',
          priority=1.0,
          trajectory={'obs': w.history['obs'][:]},
      )
      w.flush()

    samples = list(client.sample('q', num_samples=1, emit_timesteps=False))
    np.testing.assert_allclose(
        np.asarray(samples[0].data[0]), [[0.0], [1.0], [2.0]])

  def test_in_process_mutate_and_reset(self):
    server = _make_server()
    client = server.in_process_client

    with client.trajectory_writer(table='t', num_keep_alive_refs=1) as w:
      w.append({'obs': np.array([42.0], dtype=np.float32)})
      w.create_item(
          table='t', priority=1.0, trajectory={'obs': w.history['obs'][:]})
      w.flush()

    info = client.server_info()
    self.assertIn('t', info)
    self.assertEqual(info['t'].max_size, 10)

    client.reset('t')
    info_after = client.server_info()
    self.assertEqual(info_after['t'].current_size, 0)


class InProcessSelectorTest(absltest.TestCase):

  def test_uniform_replay(self):
    server = _make_server(
        table_name='u', max_size=50, min_size=1,
        sampler=reverb.selectors.Uniform())
    client = server.in_process_client

    values = [float(i) for i in range(5)]
    for v in values:
      _insert_one(client, 'u', np.array([v], dtype=np.float32))

    seen = set()
    for sample in client.sample('u', num_samples=20, emit_timesteps=False):
      seen.add(float(np.asarray(sample.data[0]).reshape(-1)[0]))
    self.assertTrue(seen.issubset(set(values)), seen)
    self.assertGreaterEqual(len(seen), 3, seen)

  def test_lifo_replay(self):
    server = _make_server(
        table_name='l', max_size=10, min_size=1,
        sampler=reverb.selectors.Lifo(),
        max_times_sampled=1)
    client = server.in_process_client

    for i in range(4):
      _insert_one(client, 'l', np.array([float(i)], dtype=np.float32))

    order = [
        float(np.asarray(sample.data[0]).reshape(-1)[0])
        for sample in client.sample('l', num_samples=4, emit_timesteps=False)
    ]
    self.assertEqual(order, [3.0, 2.0, 1.0, 0.0])

  def test_prioritized_replay(self):
    server = _make_server(
        table_name='p', max_size=20, min_size=1,
        sampler=reverb.selectors.Prioritized(1.0))
    client = server.in_process_client

    _insert_one(client, 'p', np.array([0.0], dtype=np.float32), priority=0.1)
    _insert_one(client, 'p', np.array([1.0], dtype=np.float32), priority=100.0)

    counts = {0.0: 0, 1.0: 0}
    for sample in client.sample('p', num_samples=100, emit_timesteps=False):
      val = float(np.asarray(sample.data[0]).reshape(-1)[0])
      counts[val] += 1
    self.assertGreater(counts[1.0], counts[0.0], counts)


class InProcessDtypesTest(absltest.TestCase):

  def test_multiple_dtypes(self):
    server = _make_server(table_name='d', max_size=10, min_size=1,
                          max_times_sampled=1)
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
      _insert_one(client, 'd', arr)

    samples = list(client.sample('d', num_samples=len(cases), emit_timesteps=False))
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


class InProcessRateLimiterTest(absltest.TestCase):

  def test_rate_limiter_min_size(self):
    server = _make_server(
        table_name='r', max_size=20, min_size=5,
        sampler=reverb.selectors.Fifo())
    client = server.in_process_client

    for i in range(5):
      _insert_one(client, 'r', np.array([float(i)], dtype=np.float32))

    samples = list(client.sample('r', num_samples=5, emit_timesteps=False))
    self.assertLen(samples, 5)

  def test_rate_limiter_timeout(self):
    server = _make_server(
        table_name='r', max_size=20, min_size=5,
        sampler=reverb.selectors.Fifo())
    client = server.in_process_client

    for i in range(2):
      _insert_one(client, 'r', np.array([float(i)], dtype=np.float32))

    start = time.time()
    with self.assertRaises(errors.DeadlineExceededError):
      list(client.sample('r', num_samples=1, timeout_ms=500, emit_timesteps=False))
    elapsed = time.time() - start
    self.assertLess(elapsed, 5.0, f'timeout took too long: {elapsed:.1f}s')

  def test_rate_limiter_timeout_none_compatible(self):
    server = _make_server(
        table_name='n', max_size=10, min_size=1,
        sampler=reverb.selectors.Fifo())
    client = server.in_process_client
    _insert_one(client, 'n', np.array([42.0], dtype=np.float32))
    samples = list(client.sample('n', num_samples=1, emit_timesteps=False))
    self.assertLen(samples, 1)


class InProcessCheckpointTest(absltest.TestCase):

  def test_checkpoint_save_load(self):
    root = tempfile.mkdtemp()
    values = [float(i) for i in range(3)]

    table = lambda: reverb.Table(
        name='c', sampler=reverb.selectors.Fifo(),
        remover=reverb.selectors.Fifo(), max_size=10,
        max_times_sampled=1,
        rate_limiter=reverb.rate_limiters.MinSize(1))
    server_a = reverb.Server(
        tables=[table()],
        in_process=True,
        checkpointer=checkpointers.DefaultCheckpointer(path=root))
    client_a = server_a.in_process_client
    for v in values:
      _insert_one(client_a, 'c', np.array([v], dtype=np.float32))

    ckpt_path = client_a.checkpoint()
    self.assertTrue(ckpt_path and os.path.isdir(ckpt_path), ckpt_path)
    for name in ('tables.ckpt', 'items.ckpt', 'chunks.ckpt', 'DONE'):
      self.assertTrue(os.path.exists(os.path.join(ckpt_path, name)), name)

    del server_a

    server_b = reverb.Server(
        tables=[table()],
        in_process=True,
        checkpointer=checkpointers.DefaultCheckpointer(path=root))
    client_b = server_b.in_process_client

    restored = [
        float(np.asarray(sample.data[0]).reshape(-1)[0])
        for sample in client_b.sample('c', num_samples=len(values), emit_timesteps=False)
    ]
    self.assertEqual(restored, values)


class InProcessSignatureUnpackTest(absltest.TestCase):
  """LocalClient.sample unpack_as_table_signature support."""

  def test_unpack_returns_structured_data(self):
    sig = {
        'obs': signature_codec.TensorSpec([None, 1], np.float32, 'obs'),
        'action': signature_codec.TensorSpec([None], np.int32, 'action'),
    }
    server = reverb.Server(
        tables=[reverb.Table(
            name='t', sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(), max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1), signature=sig)],
        in_process=True)
    client = server.in_process_client

    # 写入一个 3 步轨迹
    with client.trajectory_writer(table='t', num_keep_alive_refs=3) as w:
      for i in range(3):
        w.append({
            'obs': np.array([float(i)], dtype=np.float32),
            'action': np.array(i, dtype=np.int32),
        })
      w.create_item(table='t', priority=1.0, trajectory={
          'obs': w.history['obs'][:],
          'action': w.history['action'][:],
        })
      w.flush()

    # unpack_as_table_signature=True -> data 是 dict
    sample = next(client.sample('t', num_samples=1,
                                emit_timesteps=False,
                                unpack_as_table_signature=True))
    self.assertIsInstance(sample.data, dict)
    self.assertEqual(set(sample.data.keys()), {'obs', 'action'})
    np.testing.assert_array_equal(
        sample.data['obs'], [[0.], [1.], [2.]])
    np.testing.assert_array_equal(sample.data['action'], [0, 1, 2])

  def test_unpack_false_returns_flat(self):
    sig = {
        'obs': signature_codec.TensorSpec([None, 1], np.float32, 'obs'),
    }
    server = reverb.Server(
        tables=[reverb.Table(
            name='t', sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(), max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1), signature=sig)],
        in_process=True)
    client = server.in_process_client

    with client.trajectory_writer(table='t', num_keep_alive_refs=1) as w:
      w.append({'obs': np.array([42.0], dtype=np.float32)})
      w.create_item(table='t', priority=1.0,
                    trajectory={'obs': w.history['obs'][:]})
      w.flush()

    # unpack_as_table_signature=False -> data 是 flat list
    sample = next(client.sample('t', num_samples=1,
                                emit_timesteps=False,
                                unpack_as_table_signature=False))
    self.assertIsInstance(sample.data, list)
    np.testing.assert_array_equal(
        np.asarray(sample.data[0]), [[42.0]])

  def test_unpack_table_without_signature_returns_flat(self):
    # 表没声明 signature -> 即使 unpack=True 也是 flat
    server = reverb.Server(
        tables=[reverb.Table(
            name='t', sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(), max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1))],
        in_process=True)
    client = server.in_process_client

    with client.trajectory_writer(table='t', num_keep_alive_refs=1) as w:
      w.append({'v': np.array([1.0], dtype=np.float32)})
      w.create_item(table='t', priority=1.0,
                    trajectory={'v': w.history['v'][:]})
      w.flush()

    sample = next(client.sample('t', num_samples=1,
                                emit_timesteps=False,
                                unpack_as_table_signature=True))
    # signature=None -> flat list
    self.assertIsInstance(sample.data, list)

  def test_unpack_unknown_table_raises(self):
    server = reverb.Server(
        tables=[reverb.Table(
            name='t', sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(), max_size=10,
            rate_limiter=reverb.rate_limiters.MinSize(1))],
        in_process=True)
    client = server.in_process_client
    with self.assertRaises(ValueError):
      next(client.sample('nonexistent', num_samples=1,
                        unpack_as_table_signature=True,
                        timeout_ms=100))


if __name__ == '__main__':
  absltest.main()
