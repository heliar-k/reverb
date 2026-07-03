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
plain numpy arrays, without TensorFlow. This is the path surfaced by Task 15.
"""

import numpy as np
import reverb


def _make_server(table_name='t', max_size=10, min_size=1):
  return reverb.Server(
      tables=[
          reverb.Table(
              name=table_name,
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              max_size=max_size,
              rate_limiter=reverb.rate_limiters.MinSize(min_size),
          )
      ],
      in_process=True,
  )


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


if __name__ == '__main__':
  test_in_process_write_sample()
  test_in_process_multiple_steps()
  test_in_process_mutate_and_reset()
  print('PASS')
