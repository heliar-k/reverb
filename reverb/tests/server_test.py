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

"""Exhaustive tests for reverb.server (Server + Table)."""

import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from reverb import item_selectors
from reverb import pybind
from reverb import rate_limiters
from reverb import server
from reverb import signature_codec

TABLE_NAME = 'table'


class ServerTest(absltest.TestCase):

  def test_in_process_client(self):
    my_server = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Prioritized(1),
                remover=item_selectors.Fifo(),
                max_size=100,
                rate_limiter=rate_limiters.MinSize(2)),
        ])
    my_client = my_server.localhost_client()
    my_client.reset(TABLE_NAME)
    del my_client
    my_server.stop()

  def test_duplicate_priority_table_name(self):
    with self.assertRaises(ValueError):
      server.Server(
          tables=[
              server.Table(
                  name='test',
                  sampler=item_selectors.Prioritized(1),
                  remover=item_selectors.Fifo(),
                  max_size=100,
                  rate_limiter=rate_limiters.MinSize(2)),
              server.Table(
                  name='test',
                  sampler=item_selectors.Prioritized(2),
                  remover=item_selectors.Fifo(),
                  max_size=200,
                  rate_limiter=rate_limiters.MinSize(1))
          ])

  def test_no_priority_table_provided(self):
    with self.assertRaises(ValueError):
      server.Server(tables=[])

  def test_can_sample(self):
    table = server.Table(
        name=TABLE_NAME,
        sampler=item_selectors.Prioritized(1),
        remover=item_selectors.Fifo(),
        max_size=100,
        max_times_sampled=1,
        rate_limiter=rate_limiters.MinSize(2))
    my_server = server.Server(tables=[table])
    my_client = my_server.localhost_client()
    self.assertFalse(table.can_sample(1))
    self.assertTrue(table.can_insert(1))
    my_client.insert(1, {TABLE_NAME: 1.0})
    self.assertFalse(table.can_sample(1))
    my_client.insert(1, {TABLE_NAME: 1.0})
    for _ in range(100):
      if table.info.current_size == 2:
        break
      time.sleep(0.01)
    self.assertEqual(table.info.current_size, 2)
    self.assertTrue(table.can_sample(2))
    # TODO(b/153258711): This should return False since max_times_sampled=1.
    self.assertTrue(table.can_sample(3))
    del my_client
    my_server.stop()

  def test_in_process_client_raises_when_not_in_process(self):
    # gRPC server (in_process=False) must not expose in_process_client.
    srv = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=10,
                rate_limiter=rate_limiters.MinSize(1)),
        ])
    try:
      with self.assertRaisesRegex(ValueError, 'in_process=True'):
        _ = srv.in_process_client
    finally:
      srv.stop()

  def test_localhost_client_raises_in_in_process_mode(self):
    srv = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=10,
                rate_limiter=rate_limiters.MinSize(1)),
        ],
        in_process=True)
    with self.assertRaisesRegex(ValueError, 'in_process mode'):
      srv.localhost_client()

  def test_in_process_client_available_in_in_process_mode(self):
    srv = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=10,
                rate_limiter=rate_limiters.MinSize(1)),
        ],
        in_process=True)
    # in_process_client is a property that returns the LocalClient.
    client = srv.in_process_client
    self.assertIsNotNone(client)
    info = client.server_info()
    self.assertIn(TABLE_NAME, info)

  def test_stop_is_idempotent(self):
    srv = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=10,
                rate_limiter=rate_limiters.MinSize(1)),
        ])
    srv.stop()
    # Calling stop again should not raise (the guard checks _server is None).
    srv.stop()

  def test_in_process_port_is_none(self):
    srv = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=10,
                rate_limiter=rate_limiters.MinSize(1)),
        ],
        in_process=True)
    self.assertIsNone(srv.port)

  def test_grpc_port_assigned(self):
    srv = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=10,
                rate_limiter=rate_limiters.MinSize(1)),
        ])
    try:
      self.assertIsNotNone(srv.port)
      self.assertGreater(srv.port, 0)
    finally:
      srv.stop()

  def test_default_checkpointer_used_when_none(self):
    # Passing checkpointer=None picks the default TempDirCheckpointer.
    srv = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Fifo(),
                remover=item_selectors.Fifo(),
                max_size=10,
                rate_limiter=rate_limiters.MinSize(1)),
        ],
        in_process=True,
        checkpointer=None)
    self.assertIsNotNone(srv._checkpointer)


class TableTest(parameterized.TestCase):

  def _check_selector_proto(self, expected_selector, proto_msg):
    if isinstance(expected_selector, item_selectors.Uniform):
      self.assertTrue(proto_msg.HasField('uniform'))
    elif isinstance(expected_selector, item_selectors.Prioritized):
      self.assertTrue(proto_msg.HasField('prioritized'))
    elif isinstance(expected_selector, pybind.HeapSelector):
      self.assertTrue(proto_msg.HasField('heap'))
    elif isinstance(expected_selector, item_selectors.Fifo):
      self.assertTrue(proto_msg.HasField('fifo'))
    elif isinstance(expected_selector, item_selectors.Lifo):
      self.assertTrue(proto_msg.HasField('lifo'))
    else:
      raise ValueError(f'Unknown selector: {expected_selector}')

  @parameterized.product(
      sampler_fn=[
          item_selectors.Uniform,
          lambda: item_selectors.Prioritized(1.),
          item_selectors.MinHeap,
          item_selectors.MaxHeap,
          item_selectors.Fifo,
          item_selectors.Lifo
      ],
      remover_fn=[
          item_selectors.Uniform,
          lambda: item_selectors.Prioritized(1.),
          item_selectors.MinHeap,
          item_selectors.MaxHeap,
          item_selectors.Fifo,
          item_selectors.Lifo
      ],
      rate_limiter_fn=[
          lambda: rate_limiters.MinSize(10),
          lambda: rate_limiters.Queue(10),
          lambda: rate_limiters.SampleToInsertRatio(1.0, 10, 1.),
          lambda: rate_limiters.Stack(10)
      ],
      )
  def test_table_info(self, sampler_fn, remover_fn, rate_limiter_fn):
    sampler = sampler_fn()
    remover = remover_fn()
    rate_limiter = rate_limiter_fn()
    table = server.Table(
        name='table',
        sampler=sampler,
        remover=remover,
        max_size=100,
        rate_limiter=rate_limiter)
    table_info = table.info
    self.assertEqual('table', table_info.name)
    self.assertEqual(100, table_info.max_size)
    self.assertEqual(0, table_info.current_size)
    self.assertEqual(0, table_info.num_episodes)
    self.assertEqual(0, table_info.num_deleted_episodes)
    self.assertIsNone(table_info.signature)
    self._check_selector_proto(sampler, table_info.sampler_options)
    self._check_selector_proto(remover, table_info.remover_options)

  def test_table_info_signature(self):
    signature = {
        'a': signature_codec.TensorSpec([3, 3], np.float32, 'a'),
        'b': signature_codec.TensorSpec([None, 2], np.int64, 'b'),
    }
    table = server.Table(
        name='table',
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=100,
        rate_limiter=rate_limiters.MinSize(10),
        signature=signature)
    self.assertEqual(signature, table.info.signature)

  def test_replace(self):
    table = server.Table(
        name='table',
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=100,
        rate_limiter=rate_limiters.MinSize(10))
    rl_info = table.info.rate_limiter_info
    new_rate_limiter = rate_limiters.RateLimiter(
        samples_per_insert=rl_info.samples_per_insert,
        min_size_to_sample=1,
        min_diff=rl_info.min_diff,
        max_diff=rl_info.max_diff)
    new_table = table.replace(rate_limiter=new_rate_limiter)
    self.assertEqual(new_table.name, table.name)
    self.assertEqual(new_table.info.rate_limiter_info.min_size_to_sample, 1)

  def test_replace_overrides_name(self):
    table = server.Table(
        name='table',
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=100,
        rate_limiter=rate_limiters.MinSize(10))
    new_table = table.replace(name='renamed')
    self.assertEqual(new_table.name, 'renamed')

  def test_replace_overrides_max_size(self):
    table = server.Table(
        name='table',
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=100,
        rate_limiter=rate_limiters.MinSize(10))
    new_table = table.replace(max_size=50)
    self.assertEqual(new_table.info.max_size, 50)

  def test_replace_overrides_sampler(self):
    table = server.Table(
        name='table',
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=100,
        rate_limiter=rate_limiters.MinSize(10))
    new_sampler = item_selectors.Lifo()
    new_table = table.replace(sampler=new_sampler)
    self.assertTrue(new_table.info.sampler_options.lifo)

  def test_replace_overrides_signature(self):
    table = server.Table(
        name='table',
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=100,
        rate_limiter=rate_limiters.MinSize(10))
    self.assertIsNone(table.info.signature)
    sig = {'a': signature_codec.TensorSpec([2], np.float32, 'a')}
    new_table = table.replace(signature=sig)
    self.assertEqual(new_table.info.signature, sig)

  def test_queue_classmethod(self):
    table = server.Table.queue('q', 10)
    info = table.info
    self.assertEqual(info.name, 'q')
    self.assertEqual(info.max_size, 10)
    self.assertTrue(info.sampler_options.fifo)
    self.assertTrue(info.remover_options.fifo)

  def test_stack_classmethod(self):
    table = server.Table.stack('s', 10)
    info = table.info
    self.assertEqual(info.name, 's')
    self.assertEqual(info.max_size, 10)
    self.assertTrue(info.sampler_options.lifo)
    self.assertTrue(info.remover_options.lifo)

  def test_queue_with_signature(self):
    sig = {'a': signature_codec.TensorSpec([2], np.float32, 'a')}
    table = server.Table.queue('q', 10, signature=sig)
    self.assertEqual(table.info.signature, sig)

  def test_stack_with_signature(self):
    sig = {'a': signature_codec.TensorSpec([2], np.float32, 'a')}
    table = server.Table.stack('s', 10, signature=sig)
    self.assertEqual(table.info.signature, sig)


class TableConstructorValidationTest(absltest.TestCase):

  def test_empty_name_raises(self):
    with self.assertRaises(ValueError):
      server.Table(
          name='',
          sampler=item_selectors.Fifo(),
          remover=item_selectors.Fifo(),
          max_size=10,
          rate_limiter=rate_limiters.MinSize(1))

  def test_max_size_zero_raises(self):
    with self.assertRaises(ValueError):
      server.Table(
          name='t',
          sampler=item_selectors.Fifo(),
          remover=item_selectors.Fifo(),
          max_size=0,
          rate_limiter=rate_limiters.MinSize(1))

  def test_max_size_negative_raises(self):
    with self.assertRaises(ValueError):
      server.Table(
          name='t',
          sampler=item_selectors.Fifo(),
          remover=item_selectors.Fifo(),
          max_size=-1,
          rate_limiter=rate_limiters.MinSize(1))

  def test_unsupported_signature_leaf_raises(self):
    with self.assertRaises(ValueError):
      server.Table(
          name='t',
          sampler=item_selectors.Fifo(),
          remover=item_selectors.Fifo(),
          max_size=10,
          rate_limiter=rate_limiters.MinSize(1),
          signature={'a': 'not a TensorSpec'})

  def test_repr_returns_string(self):
    table = server.Table(
        name='t',
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=10,
        rate_limiter=rate_limiters.MinSize(1))
    self.assertIsInstance(repr(table), str)


if __name__ == '__main__':
  absltest.main()
