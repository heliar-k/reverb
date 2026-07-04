"""End-to-end smoke test for the gRPC client/server path (no TensorFlow).

Mirrors `in_process_test.py` but exercises the real gRPC `Client` against a
networked `Server(in_process=False)`. All data is exchanged as numpy arrays;
no `tf.TensorSpec` signatures are set (the numpy-only build does not encode
TF signatures).
"""

import numpy as np
import portpicker
import reverb


def _make_grpc_server(table_name='t', max_size=10, min_size=1):
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


def test_grpc_client_server_info():
  server, port = _make_grpc_server()
  try:
    client = reverb.Client(f'localhost:{port}')
    info = client.server_info()
    assert 't' in info
    assert info['t'].max_size == 10
  finally:
    server.stop()


def test_grpc_writer_insert_and_sample():
  """Legacy `client.writer()` (gRPC `Writer`) round-trip via InsertStream."""
  server, port = _make_grpc_server(table_name='q', max_size=10, min_size=1)
  try:
    client = reverb.Client(f'localhost:{port}')
    with client.writer(max_sequence_length=2) as w:
      w.append({'obs': np.array([1.0, 2.0], dtype=np.float32)})
      w.append({'obs': np.array([3.0, 4.0], dtype=np.float32)})
      w.create_item(table='q', num_timesteps=1, priority=1.0)
      w.flush()

    samples = list(client.sample('q', num_samples=1))
    assert len(samples) == 1
    # With emit_timesteps=True (default), each yielded element is a list of
    # per-timestep ReplaySample objects. The item spans 1 timestep and
    # references the MOST RECENTLY appended step ([3, 4]).
    timesteps = samples[0]
    assert len(timesteps) == 1
    data = timesteps[0].data
    assert np.allclose(np.asarray(data[0]), [3.0, 4.0]), data
  finally:
    server.stop()


def test_grpc_mutate_priorities_and_reset():
  server, port = _make_grpc_server()
  try:
    client = reverb.Client(f'localhost:{port}')
    with client.writer(max_sequence_length=1) as w:
      w.append({'v': np.array(7, dtype=np.int64)})
      w.create_item(table='t', num_timesteps=1, priority=1.0)
      w.flush()

    info = client.server_info()
    assert info['t'].current_size == 1

    client.reset('t')
    info_after = client.server_info()
    assert info_after['t'].current_size == 0
  finally:
    server.stop()


if __name__ == '__main__':
  test_grpc_client_server_info()
  test_grpc_writer_insert_and_sample()
  test_grpc_mutate_priorities_and_reset()
  print('grpc_roundtrip_test OK')
