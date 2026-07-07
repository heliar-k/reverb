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

"""Tests for python client."""

import collections
import multiprocessing.dummy as multithreading
import pickle
import time

from absl.testing import absltest
import numpy as np
import portpicker
from reverb import client
from reverb import errors
from reverb import item_selectors
from reverb import rate_limiters
from reverb import server
import tree

TABLE_NAME = 'table'
NESTED_SIGNATURE_TABLE_NAME = 'nested_signature_table'
SIMPLE_QUEUE_NAME = 'simple_queue'
# ponytail: 原 QUEUE_SIGNATURE 用 tf.TensorSpec 构造 table signature;TF 已移除
# 后 numpy-only 模式不支持构造 signature,全部置 None。保留表名与结构以便
# 用例名称语义不变,signature 相关断言改为 assertIsNone。
QUEUE_SIGNATURE = None


class ClientTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tables = [
        server.Table(
            name=TABLE_NAME,
            sampler=item_selectors.Prioritized(1),
            remover=item_selectors.Fifo(),
            max_size=1000,
            rate_limiter=rate_limiters.MinSize(3),
            # ponytail: 原 signature=tf.TensorSpec(dtype=tf.int64, shape=[]);
            # TF 移除后 numpy-only 不支持构造 signature,置 None。
            signature=None,
        ),
        server.Table.queue(
            name=NESTED_SIGNATURE_TABLE_NAME,
            max_size=10,
            signature=QUEUE_SIGNATURE,
        ),
        server.Table.queue(SIMPLE_QUEUE_NAME, 10),
    ]
    cls.server = server.Server(tables=cls.tables)
    cls.client = cls.server.localhost_client()

  def tearDown(self):
    self.client.reset(TABLE_NAME)
    self.client.reset(NESTED_SIGNATURE_TABLE_NAME)
    self.client.reset(SIMPLE_QUEUE_NAME)
    super().tearDown()

  @classmethod
  def tearDownClass(cls):
    cls.server.stop()
    super().tearDownClass()

  def wait_for_table_size(self, size):
    for _ in range(100):
      if self.tables[0].info.current_size == size:
        break
      time.sleep(0.01)
    self.assertEqual(self.tables[0].info.current_size, size)

  def _get_sample_frequency(self, n=10000):
    keys = [sample[0].info.key for sample in self.client.sample(TABLE_NAME, n)]
    counter = collections.Counter(keys)
    return [count / n for _, count in counter.most_common()]

  def test_sample_sets_table_size(self):
    for i in range(1, 11):
      self.client.insert(i, {TABLE_NAME: 1.0})
      if i >= 3:
        sample = next(self.client.sample(TABLE_NAME, 1))[0]
        self.assertEqual(sample.info.table_size, i)
    self.wait_for_table_size(10)

  def test_sample_sets_probability(self):
    for i in range(1, 11):
      self.client.insert(i, {TABLE_NAME: 1.0})
      if i >= 3:
        sample = next(self.client.sample(TABLE_NAME, 1))[0]
        self.assertAlmostEqual(sample.info.probability, 1.0 / i, 0.01)

  def test_sample_sets_priority(self):
    # Set the test context by manually mutating priorities to known ones.
    for i in range(10):
      self.client.insert(i, {TABLE_NAME: 1000.0})
    self.wait_for_table_size(10)

    def _sample_priorities(n=100):
      return {
          sample[0].info.key: sample[0].info.priority
          for sample in self.client.sample(TABLE_NAME, n)
      }

    original_priorities = _sample_priorities(n=100)
    self.assertNotEmpty(original_priorities)
    self.assertSequenceAlmostEqual([1000.0] * len(original_priorities),
                                   original_priorities.values())
    expected_priorities = {
        key: float(i) for i, key in enumerate(original_priorities)
    }
    self.client.mutate_priorities(TABLE_NAME, updates=expected_priorities)

    # Resample and check priorities.
    sampled_priorities = _sample_priorities(n=100)
    self.assertNotEmpty(sampled_priorities)
    for key, priority in sampled_priorities.items():
      if key in expected_priorities:
        self.assertAlmostEqual(expected_priorities[key], priority)

  def test_insert_raises_if_priorities_empty(self):
    with self.assertRaises(ValueError):
      self.client.insert([1], {})

  def test_insert(self):
    self.client.insert(1, {TABLE_NAME: 1.0})  # This should be sampled often.
    self.client.insert(2, {TABLE_NAME: 0.1})  # This should be sampled rarely.
    self.client.insert(3, {TABLE_NAME: 0.0})  # This should never be sampled.
    self.wait_for_table_size(3)

    freqs = self._get_sample_frequency()

    self.assertLen(freqs, 2)
    self.assertAlmostEqual(freqs[0], 0.9, delta=0.05)
    self.assertAlmostEqual(freqs[1], 0.1, delta=0.05)

  def test_writer_raises_if_max_sequence_length_lt_1(self):
    with self.assertRaises(ValueError):
      self.client.writer(0)

  def test_writer_raises_if_chunk_length_lt_1(self):
    self.client.writer(2, chunk_length=1)  # Should be fine.

    for chunk_length in [0, -1]:
      with self.assertRaises(ValueError):
        self.client.writer(2, chunk_length=chunk_length)

  def test_writer_raises_if_chunk_length_gt_max_sequence_length(self):
    self.client.writer(2, chunk_length=1)  # lt should be fine.
    self.client.writer(2, chunk_length=2)  # eq should be fine.

    with self.assertRaises(ValueError):
      self.client.writer(2, chunk_length=3)

  def test_writer_raises_if_max_in_flight_items_lt_1(self):
    self.client.writer(1, max_in_flight_items=1)
    self.client.writer(1, max_in_flight_items=2)

    with self.assertRaises(ValueError):
      self.client.writer(1, max_in_flight_items=-1)

  def test_writer_works_with_no_retries(self):
    # If the server responds correctly, the writer ignores the no retries arg.
    writer = self.client.writer(2)
    writer.append([0])
    writer.create_item(TABLE_NAME, 1, 1.0)
    writer.close(retry_on_unavailable=False)

  def test_writer(self):
    with self.client.writer(2) as writer:
      writer.append([0])
      writer.create_item(TABLE_NAME, 1, 1.0)
      writer.append([1])
      writer.create_item(TABLE_NAME, 2, 1.0)
      writer.append([2])
      writer.create_item(TABLE_NAME, 1, 1.0)
      writer.append_sequence([np.array([3, 4])])
      writer.create_item(TABLE_NAME, 2, 1.0)

    freqs = self._get_sample_frequency()
    self.assertLen(freqs, 4)
    for freq in freqs:
      self.assertAlmostEqual(freq, 0.25, delta=0.05)

  def test_write_and_sample_different_shapes_and_dtypes(self):
    trajectories = [
        np.ones([], np.int64),
        np.ones([2, 2], np.float32),
        np.ones([3, 3], np.int32),
    ]
    for trajectory in trajectories:
      self.client.insert(trajectory, {SIMPLE_QUEUE_NAME: 1.0})

    for i, [sample] in enumerate(self.client.sample(SIMPLE_QUEUE_NAME, 3)):
      np.testing.assert_array_equal(trajectories[i], sample.data[0])

  def test_mutate_priorities_update(self):
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})

    before = self._get_sample_frequency()
    self.assertLen(before, 3)
    for freq in before:
      self.assertAlmostEqual(freq, 0.33, delta=0.05)

    key = next(self.client.sample(TABLE_NAME, 1))[0].info.key
    self.client.mutate_priorities(TABLE_NAME, updates={key: 0.5})

    after = self._get_sample_frequency()
    self.assertLen(after, 3)
    self.assertAlmostEqual(after[0], 0.4, delta=0.05)
    self.assertAlmostEqual(after[1], 0.4, delta=0.05)
    self.assertAlmostEqual(after[2], 0.2, delta=0.05)

  def test_mutate_priorities_delete(self):
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})

    before = self._get_sample_frequency()
    self.assertLen(before, 4)

    key = next(self.client.sample(TABLE_NAME, 1))[0].info.key
    self.client.mutate_priorities(TABLE_NAME, deletes=[key])

    after = self._get_sample_frequency()
    self.assertLen(after, 3)

  def test_reset(self):
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})

    keys_before = set(
        sample[0].info.key for sample in self.client.sample(TABLE_NAME, 1000))
    self.assertLen(keys_before, 3)

    self.client.reset(TABLE_NAME)

    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})

    keys_after = set(
        sample[0].info.key for sample in self.client.sample(TABLE_NAME, 1000))
    self.assertLen(keys_after, 3)

    self.assertTrue(keys_after.isdisjoint(keys_before))

  def test_server_info(self):
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    list(self.client.sample(TABLE_NAME, 1))
    server_info = self.client.server_info()
    self.assertLen(server_info, 3)

    self.assertIn(TABLE_NAME, server_info)
    table = server_info[TABLE_NAME]
    self.assertEqual(table.current_size, 3)
    self.assertEqual(table.num_unique_samples, 1)
    self.assertEqual(table.max_size, 1000)
    self.assertEqual(table.sampler_options.prioritized.priority_exponent, 1)
    self.assertTrue(table.remover_options.fifo)
    # ponytail: signature 原 == tf.TensorSpec(...);numpy-only 置 None。
    self.assertIsNone(table.signature)

    self.assertIn(NESTED_SIGNATURE_TABLE_NAME, server_info)
    queue = server_info[NESTED_SIGNATURE_TABLE_NAME]
    self.assertEqual(queue.current_size, 0)
    self.assertEqual(queue.num_unique_samples, 0)
    self.assertEqual(queue.max_size, 10)
    self.assertTrue(queue.sampler_options.fifo)
    self.assertTrue(queue.remover_options.fifo)
    # ponytail: queue.signature 原 == QUEUE_SIGNATURE(tf.TensorSpec dict);置 None。
    self.assertIsNone(queue.signature)

    self.assertIn(SIMPLE_QUEUE_NAME, server_info)
    info = server_info[SIMPLE_QUEUE_NAME]
    self.assertEqual(info.current_size, 0)
    self.assertEqual(info.num_unique_samples, 0)
    self.assertEqual(info.max_size, 10)
    self.assertTrue(info.sampler_options.fifo)
    self.assertTrue(info.remover_options.fifo)
    self.assertIsNone(info.signature)

  def test_sample_trajectory_with_signature(self):
    # ponytail: 该表 signature 原为 QUEUE_SIGNATURE(tf.TensorSpec dict),
    # unpack_as_table_signature=True 会按结构拆包成 dict。TF 移除后
    # signature=None,unpack 退化为 flat 列表(同 without_signature 路径)。
    # 降级:断言 flat 行为以保留用例(不删)。
    # 恢复路径:重新支持 signature 后改回 dict 断言。
    with self.client.trajectory_writer(3) as writer:
      for _ in range(3):
        writer.append({
            'a': np.ones([], np.int64),
            'b': np.ones([2, 2], np.float32),
        })

      writer.create_item(
          table=NESTED_SIGNATURE_TABLE_NAME,
          priority=1.0,
          trajectory={
              'a': writer.history['a'][:],
              'b': writer.history['b'][:],
          })

    sample = next(self.client.sample(NESTED_SIGNATURE_TABLE_NAME,
                                     emit_timesteps=False,
                                     unpack_as_table_signature=True))

    # signature=None -> flat 数据,每个元素代表整列。
    want = [np.ones([3], np.int64), np.ones([3, 2, 2], np.float32)]
    tree.map_structure(np.testing.assert_array_equal, sample.data, want)

    # The info fields should all be scalars (i.e not batched by time).
    self.assertIsInstance(sample.info.key, int)
    self.assertIsInstance(sample.info.probability, float)
    self.assertIsInstance(sample.info.table_size, int)
    self.assertIsInstance(sample.info.priority, float)

  def test_sample_trajectory_without_signature(self):
    with self.client.trajectory_writer(3) as writer:
      for _ in range(3):
        writer.append({
            'a': np.ones([], np.int64),
            'b': np.ones([2, 2], np.float32),
        })

      writer.create_item(
          table=SIMPLE_QUEUE_NAME,
          priority=1.0,
          trajectory={
              'a': writer.history['a'][:],
              'b': writer.history['b'][:],
          })

    sample = next(self.client.sample(SIMPLE_QUEUE_NAME,
                                     emit_timesteps=False,
                                     unpack_as_table_signature=True))

    # The data should be flat as the table has no signature. Each element within
    # the flat data should represent the entire column (i.e not just one step).
    want = [np.ones([3], np.int64), np.ones([3, 2, 2], np.float32)]
    tree.map_structure(np.testing.assert_array_equal, sample.data, want)

    # The info fields should all be scalars (i.e not batched by time).
    self.assertIsInstance(sample.info.key, int)
    self.assertIsInstance(sample.info.probability, float)
    self.assertIsInstance(sample.info.table_size, int)
    self.assertIsInstance(sample.info.priority, float)

  def test_sample_trajectory_as_flat_data(self):
    with self.client.trajectory_writer(3) as writer:
      for _ in range(3):
        writer.append({
            'a': np.ones([], np.int64),
            'b': np.ones([2, 2], np.float32),
        })

      writer.create_item(
          table=NESTED_SIGNATURE_TABLE_NAME,
          priority=1.0,
          trajectory={
              'a': writer.history['a'][:],
              'b': writer.history['b'][:],
          })

    sample = next(self.client.sample(NESTED_SIGNATURE_TABLE_NAME,
                                     emit_timesteps=False,
                                     unpack_as_table_signature=False))

    # The table has a signature but we requested the data to be flat.
    want = [np.ones([3], np.int64), np.ones([3, 2, 2], np.float32)]
    tree.map_structure(np.testing.assert_array_equal, sample.data, want)

    # The info fields should all be scalars (i.e not batched by time).
    self.assertIsInstance(sample.info.key, int)
    self.assertIsInstance(sample.info.probability, float)
    self.assertIsInstance(sample.info.table_size, int)
    self.assertIsInstance(sample.info.priority, float)

  def test_sample_trajectory_written_with_insert(self):
    self.client.insert(np.ones([3, 3], np.int32), {SIMPLE_QUEUE_NAME: 1.0})

    sample = next(self.client.sample(SIMPLE_QUEUE_NAME,
                                     emit_timesteps=False))

    # An extra batch dimension should have been added to the inserted data as
    # it is a trajectory of length 1.
    want = [np.ones([1, 3, 3], np.int32)]
    tree.map_structure(np.testing.assert_array_equal, sample.data, want)

    # The info fields should all be scalars (i.e not batched by time).
    self.assertIsInstance(sample.info.key, int)
    self.assertIsInstance(sample.info.probability, float)
    self.assertIsInstance(sample.info.table_size, int)
    self.assertIsInstance(sample.info.priority, float)

  def test_sample_trajectory_written_with_legacy_writer(self):
    with self.client.writer(3) as writer:
      for i in range(3):
        writer.append([i, np.ones([2, 2], np.float64)])

      writer.create_item(SIMPLE_QUEUE_NAME, 3, 1.0)

    sample = next(self.client.sample(SIMPLE_QUEUE_NAME,
                                     emit_timesteps=False))

    # The time dimension should have been added to all fields.
    want = [np.array([0, 1, 2]), np.ones([3, 2, 2], np.float64)]
    tree.map_structure(np.testing.assert_array_equal, sample.data, want)

    # The info fields should all be scalars (i.e not batched by time).
    self.assertIsInstance(sample.info.key, int)
    self.assertIsInstance(sample.info.probability, float)
    self.assertIsInstance(sample.info.table_size, int)
    self.assertIsInstance(sample.info.priority, float)

  def test_server_info_timeout(self):
    try:
      # Setup a client that doesn't actually connect to anything.
      dummy_port = portpicker.pick_unused_port()
      dummy_client = client.Client(f'localhost:{dummy_port}')
      with self.assertRaises(
          errors.DeadlineExceededError,
          msg='ServerInfo call did not complete within provided timeout of 1s'):
        dummy_client.server_info(timeout=1)
    finally:
      portpicker.return_port(dummy_port)

  def test_pickle(self):
    loaded_client = pickle.loads(pickle.dumps(self.client))
    self.assertEqual(loaded_client._server_address, self.client._server_address)
    loaded_client.insert([0], {TABLE_NAME: 1.0})
    self.wait_for_table_size(1)

  def test_multithreaded_writer_using_flush(self):
    # Ensure that we don't have any errors caused by multithreaded use of
    # writers or clients.
    pool = multithreading.Pool(64)
    def _write(i):
      with self.client.writer(1) as writer:
        writer.append([i])
        # Make sure that flush before create_item doesn't create trouble.
        writer.flush()
        writer.create_item(TABLE_NAME, 1, 1.0)
        writer.flush()

    for _ in range(5):
      pool.map(_write, list(range(128)))

    self.wait_for_table_size(640)
    pool.close()
    pool.join()

  def test_multithreaded_writer_using_scope(self):
    # Ensure that we don't have any errors caused by multithreaded use of
    # writers or clients.
    pool = multithreading.Pool(64)
    def _write(i):
      with self.client.writer(1) as writer:
        writer.append([i])
        writer.create_item(TABLE_NAME, 1, 1.0)

    for _ in range(5):
      pool.map(_write, list(range(256)))

    info = self.client.server_info()[TABLE_NAME]
    self.assertEqual(info.current_size, 1000)
    pool.close()
    pool.join()

  def test_validates_trajectory_writer_config(self):
    with self.assertRaises(ValueError):
      self.client.trajectory_writer(0)

    with self.assertRaises(ValueError):
      self.client.trajectory_writer(-1)


# ponytail: 边界覆盖补充——Client/Writer/LocalClient 导出方法的穷尽分支。
class ClientBoundaryTest(ClientTest):
  """Reuses ClientTest's setUpClass server/client fixtures."""

  def test_structured_writer_empty_configs_raises(self):
    with self.assertRaises(ValueError):
      self.client.structured_writer([])

  def test_server_address_property(self):
    self.assertEqual(self.client.server_address,
                     f'localhost:{self.server.port}')

  def test_client_repr(self):
    r = repr(self.client)
    self.assertIn('Client', r)
    self.assertIn('localhost', r)

  def test_pickle_round_trip_address(self):
    import pickle
    loaded = pickle.loads(pickle.dumps(self.client))
    self.assertEqual(loaded.server_address, self.client.server_address)

  def test_checkpoint_returns_path(self):
    path = self.client.checkpoint()
    self.assertIsInstance(path, str)
    self.assertTrue(path)

  def test_writer_repr(self):
    w = self.client.writer(2)
    r = repr(w)
    self.assertIn('closed', r)
    w.close()


class WriterBoundaryTest(absltest.TestCase):
  """Writer lifecycle / validation edges not covered by ClientTest."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=100,
                rate_limiter=rate_limiters.MinSize(1)),
        ])
    cls.client = cls._server.localhost_client()

  @classmethod
  def tearDownClass(cls):
    cls._server.stop()
    super().tearDownClass()

  def test_close_twice_raises(self):
    w = self.client.writer(2)
    w.close()
    with self.assertRaises(ValueError):
      w.close()

  def test_create_item_zero_timesteps_raises(self):
    w = self.client.writer(2)
    with self.assertRaises(ValueError):
      w.create_item(TABLE_NAME, 0, 1.0)
    w.close()

  def test_create_item_negative_timesteps_raises(self):
    w = self.client.writer(2)
    with self.assertRaises(ValueError):
      w.create_item(TABLE_NAME, -1, 1.0)
    w.close()

  def test_context_manager_closes(self):
    with self.client.writer(2) as w:
      w.append([1])
    # Reusing a closed writer via the context manager protocol raises.
    with self.assertRaises(ValueError):
      with w:
        pass

  def test_flush_is_idempotent(self):
    w = self.client.writer(2)
    w.flush()
    w.flush()
    w.close()

  def test_append_sequence(self):
    with self.client.writer(3) as w:
      w.append_sequence([np.array([1, 2, 3])])
      w.create_item(TABLE_NAME, 2, 1.0)


class LocalClientBoundaryTest(absltest.TestCase):
  """LocalClient (in-process numpy mode) specific edges."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=100,
                rate_limiter=rate_limiters.MinSize(1)),
        ],
        in_process=True)
    cls.client = cls._server.in_process_client

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()

  def test_repr(self):
    self.assertIn('LocalClient', repr(self.client))

  def test_trajectory_writer_invalid_refs_raises(self):
    with self.assertRaises(ValueError):
      self.client.trajectory_writer(table=TABLE_NAME, num_keep_alive_refs=0)

  def test_trajectory_writer_negative_refs_raises(self):
    with self.assertRaises(ValueError):
      self.client.trajectory_writer(table=TABLE_NAME, num_keep_alive_refs=-1)

  def test_trajectory_writer_max_chunk_length(self):
    w = self.client.trajectory_writer(
        table=TABLE_NAME, num_keep_alive_refs=2, max_chunk_length=1)
    w.append({'v': np.array([1.0], dtype=np.float32)})
    w.end_episode()

  def test_new_sampler_timeout_raises(self):
    # MinSize(5) but no items: sampling must block and raise DeadlineExceeded.
    srv = server.Server(
        tables=[
            server.Table(
                name='blocked',
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=10,
                rate_limiter=rate_limiters.MinSize(5)),
        ],
        in_process=True)
    c = srv.in_process_client
    with self.assertRaises(errors.DeadlineExceededError):
      list(c.sample('blocked', num_samples=1, timeout_ms=200))

  def test_mutate_priorities_no_args_is_noop(self):
    # No updates/deletes -> should not raise.
    self.client.mutate_priorities(TABLE_NAME)

  def test_reset_unknown_table(self):
    # reset on an unknown table surfaces a C++ error; assert it raises.
    with self.assertRaises(Exception):
      self.client.reset('nonexistent_table')

  def test_server_info(self):
    info = self.client.server_info()
    self.assertIn(TABLE_NAME, info)
    self.assertEqual(info[TABLE_NAME].max_size, 100)

  def test_checkpoint(self):
    path = self.client.checkpoint()
    self.assertIsInstance(path, str)
    self.assertTrue(path)


if __name__ == '__main__':
  absltest.main()
