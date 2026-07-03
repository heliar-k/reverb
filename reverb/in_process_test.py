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
plain numpy arrays, without TensorFlow. This is the path surfaced by Task 15
and broadened (more selectors / dtypes) in Task 17.
"""

import os
import tempfile

import numpy as np
import reverb
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


def test_in_process_write_sample():
  server = _make_server()
  client = server.in_process_client  # LocalClient wrapping InProcessClient

  with client.trajectory_writer(table='t', num_keep_alive_refs=1) as w:
    w.append({'obs': np.array([1.0, 2.0], dtype=np.float32)})
    w.create_item(
        table='t', priority=1.0, trajectory={'obs': w.history['obs'][:]})
    w.flush()

  samples = list(client.sample('t', num_samples=1))
  assert len(samples) == 1
  # data is a flat list of column arrays; single column -> [1, 2] with batch.
  assert np.allclose(np.asarray(samples[0].data[0]), [[1.0, 2.0]]), samples[0]


def test_in_process_multiple_steps():
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

  samples = list(client.sample('q', num_samples=1))
  assert np.allclose(np.asarray(samples[0].data[0]), [[0.0], [1.0], [2.0]]), (
      samples[0])


def test_in_process_mutate_and_reset():
  server = _make_server()
  client = server.in_process_client

  with client.trajectory_writer(table='t', num_keep_alive_refs=1) as w:
    w.append({'obs': np.array([42.0], dtype=np.float32)})
    w.create_item(
        table='t', priority=1.0, trajectory={'obs': w.history['obs'][:]})
    w.flush()

  info = client.server_info()
  assert 't' in info
  assert info['t'].max_size == 10

  client.reset('t')
  info_after = client.server_info()
  assert info_after['t'].current_size == 0


def test_uniform_replay():
  """Uniform selector: samples spread across all inserted items."""
  server = _make_server(
      table_name='u', max_size=50, min_size=1,
      sampler=reverb.selectors.Uniform())
  client = server.in_process_client

  values = [float(i) for i in range(5)]
  for v in values:
    _insert_one(client, 'u', np.array([v], dtype=np.float32))

  seen = set()
  for sample in client.sample('u', num_samples=20):
    seen.add(float(np.asarray(sample.data[0]).reshape(-1)[0]))
  # Every sample must come from the inserted set, and the sampler must spread
  # draws across items (a stuck sampler returning one item would fail here).
  assert seen.issubset(set(values)), seen
  assert len(seen) >= 3, seen


def test_lifo_replay():
  """LIFO sampler: the most-recently inserted item is drawn first."""
  # max_times_sampled=1 so each item is removed after one draw, letting us
  # observe insertion order through successive samples.
  server = _make_server(
      table_name='l', max_size=10, min_size=1,
      sampler=reverb.selectors.Lifo(),
      max_times_sampled=1)
  client = server.in_process_client

  for i in range(4):
    _insert_one(client, 'l', np.array([float(i)], dtype=np.float32))

  order = [
      float(np.asarray(sample.data[0]).reshape(-1)[0])
      for sample in client.sample('l', num_samples=4)
  ]
  assert order == [3.0, 2.0, 1.0, 0.0], order


def test_prioritized_replay():
  """Prioritized selector: high-priority item is sampled more often."""
  server = _make_server(
      table_name='p', max_size=20, min_size=1,
      sampler=reverb.selectors.Prioritized(1.0))
  client = server.in_process_client

  # Two items with a large priority gap.
  _insert_one(client, 'p', np.array([0.0], dtype=np.float32), priority=0.1)
  _insert_one(client, 'p', np.array([1.0], dtype=np.float32), priority=100.0)

  counts = {0.0: 0, 1.0: 0}
  for sample in client.sample('p', num_samples=100):
    val = float(np.asarray(sample.data[0]).reshape(-1)[0])
    counts[val] += 1
  assert counts[1.0] > counts[0.0], counts


def test_multiple_dtypes():
  """A single table round-trips several numpy dtypes with values intact."""
  # max_times_sampled=1 so each inserted item is drawn exactly once; with the
  # FIFO sampler this yields them in insertion order.
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

  samples = list(client.sample('d', num_samples=len(cases)))
  assert len(samples) == len(cases)
  for arr, sample in zip(cases, samples):
    got = np.asarray(sample.data[0]).reshape(-1)[0]
    assert got.dtype == arr.dtype, (got.dtype, arr.dtype)
    exp = arr[0]
    if np.issubdtype(arr.dtype, np.bool_):
      assert bool(got) == bool(exp), (got, exp)
    elif np.issubdtype(arr.dtype, np.floating):
      assert np.isclose(got, exp), (got, exp)
    else:
      assert int(got) == int(exp), (got, exp)


def test_rate_limiter_min_size():
  """MinSize(N): once N items are present, sampling succeeds.

  The blocking-under-N case cannot be asserted here: `LocalClient.sample`
  blocks indefinitely (no timeout argument is exposed on the numpy path), so
  a pre-threshold sample call would hang the test rather than fail it.
  # TODO: expose a timeout on LocalClient.sample to test the block.
  """
  server = _make_server(
      table_name='r', max_size=20, min_size=5,
      sampler=reverb.selectors.Fifo())
  client = server.in_process_client

  for i in range(5):
    _insert_one(client, 'r', np.array([float(i)], dtype=np.float32))

  samples = list(client.sample('r', num_samples=5))
  assert len(samples) == 5


def test_checkpoint_save_load():
  """Checkpoint save is exposed; load into a fresh in-process server is not.

  `LocalClient.checkpoint()` writes a timestamped checkpoint directory
  (tables.ckpt / items.ckpt / chunks.ckpt / DONE) under the checkpointer root.
  Restoring into a new `in_process=True` server is NOT wired up: unlike the
  gRPC path, the in-process `Server`/`InProcessClient` never calls
  `Checkpointer::LoadLatest` at construction, so a rebuilt server starts empty.
  # TODO: expose restore (LoadLatest) on the in-process path; until then only
  # save is testable end-to-end.
  """
  root = tempfile.mkdtemp()
  table = lambda: reverb.Table(
      name='c', sampler=reverb.selectors.Fifo(),
      remover=reverb.selectors.Fifo(), max_size=10,
      max_times_sampled=1,
      rate_limiter=reverb.rate_limiters.MinSize(1))
  server = reverb.Server(
      tables=[table()],
      in_process=True,
      checkpointer=checkpointers.DefaultCheckpointer(path=root))
  client = server.in_process_client

  for i in range(3):
    _insert_one(client, 'c', np.array([float(i)], dtype=np.float32))

  ckpt_path = client.checkpoint()
  assert ckpt_path and os.path.isdir(ckpt_path), ckpt_path
  # A complete checkpoint writes all four artifacts.
  for name in ('tables.ckpt', 'items.ckpt', 'chunks.ckpt', 'DONE'):
    assert os.path.exists(os.path.join(ckpt_path, name)), name


if __name__ == '__main__':
  test_in_process_write_sample()
  test_in_process_multiple_steps()
  test_in_process_mutate_and_reset()
  test_uniform_replay()
  test_lifo_replay()
  test_prioritized_replay()
  test_multiple_dtypes()
  test_rate_limiter_min_size()
  test_checkpoint_save_load()
  print('PASS')
