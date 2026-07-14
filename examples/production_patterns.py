# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production patterns for Reverb (numpy, in-process).

Demonstrates:
  - Selectors & removers: Queue, Circular Buffer, Stack (LIFO)
  - Rate limiters: SampleToInsertRatio
  - `flush(block_until_num_items=N)` — backpressure control
  - `append_sequence` — batch append for performance
  - `Writer` with `delta_encoded` / `chunk_length` — old-style streaming
  - `client.insert()` — direct single-item insert
  - `client.reset(table)` — clear a table
  - `Table.can_sample()` / `Table.can_insert()` — rate-limiter queries
  - `Table.replace()` — clone table with overrides

Run:
    python examples/production_patterns.py
"""

import numpy as np

import reverb


# ---------------------------------------------------------------------------
# Example 1: Queue and Circular Buffer (selectors + removers)
# ---------------------------------------------------------------------------
def example_1_queue_and_buffer():
    """Any selector can be used for sampling or removal. Combine with
    `max_times_sampled` and rate limiter to build queues, stacks, circular
    buffers.

    `Table.queue` is shorthand for Fifo sampler + Fifo remover +
    `max_times_sampled=1` + `Queue` rate limiter: each item is sampled
    exactly once then removed.
    """
    print("=" * 60)
    print("Example 1: Queue & Circular Buffer")
    print("=" * 60)

    server = reverb.Server(
        tables=[
            # A FIFO queue: sample once, then remove.
            reverb.Table.queue(name='my_queue', max_size=1000),
            # A circular buffer: uniform sample, FIFO remove, resampleable.
            reverb.Table(
                name='my_buffer',
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=1000,
                rate_limiter=reverb.rate_limiters.MinSize(1),
            ),
        ],
        in_process=True,
    )

    try:
        client = server.in_process_client
        print("  Tables:", list(client.server_info().keys()))

        # Insert into both tables.
        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            for i in range(3):
                writer.append({'v': np.array([i], dtype=np.float32)})

            # Queue: item will be removed after one sample.
            writer.create_item(
                table='my_queue', priority=1.0,
                trajectory={'v': writer.history['v'][-1:]},
            )
            # Buffer: items can be sampled multiple times.
            writer.create_item(
                table='my_buffer', priority=1.0,
                trajectory={'v': writer.history['v'][-1:]},
            )
            writer.flush()

        # Sample from queue: items removed after sampling.
        q_samples = list(client.sample('my_queue', num_samples=1,
                                        emit_timesteps=False))
        print(f"  Queue sampled: {np.asarray(q_samples[0].data[0])}")
        print(f"  Queue size after sample: "
              f"{client.server_info()['my_queue'].current_size}")

        # Circular buffer: items remain.
        for _ in range(2):
            b_samples = list(client.sample('my_buffer', num_samples=1,
                                           emit_timesteps=False))
            print(f"  Buffer sampled: {np.asarray(b_samples[0].data[0])}")
        print(f"  Buffer size after 2 samples: "
              f"{client.server_info()['my_buffer'].current_size}")

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 2: LIFO Stack
# ---------------------------------------------------------------------------
def example_2_stack():
    """`Table.stack` is the LIFO counterpart of `Table.queue`:
    newest item is sampled first, removed after sampling once.
    """
    print()
    print("=" * 60)
    print("Example 2: LIFO Stack")
    print("=" * 60)

    server = reverb.Server(
        tables=[reverb.Table.stack(name='my_stack', max_size=100)],
        in_process=True,
    )

    try:
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            for i in range(3):
                writer.append({'v': np.array([i], dtype=np.float32)})
                writer.create_item(
                    table='my_stack', priority=1.0,
                    trajectory={'v': writer.history['v'][-1:]},
                )
            writer.flush()

        # LIFO: newest first.
        for _ in range(3):
            sample = next(client.sample('my_stack', num_samples=1,
                                        emit_timesteps=False))
            print(f"  Popped: {np.asarray(sample.data[0]).reshape(-1)[0]:.0f}")

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 3: SampleToInsertRatio rate limiter
# ---------------------------------------------------------------------------
def example_3_sample_to_insert_ratio():
    """`SampleToInsertRatio` keeps the long-run ratio of samples to inserts
    near a target, blocking whichever side gets ahead. `timeout_ms` on `sample`
    turns a block into a `DeadlineExceededError` instead of waiting forever.
    """
    print()
    print("=" * 60)
    print("Example 3: SampleToInsertRatio")
    print("=" * 60)

    server = reverb.Server(
        tables=[
            reverb.Table(
                name='my_table',
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=1000,
                rate_limiter=reverb.rate_limiters.SampleToInsertRatio(
                    samples_per_insert=1.0,
                    min_size_to_sample=2,
                    error_buffer=1.0,
                ),
            )
        ],
        in_process=True,
    )

    try:
        client = server.in_process_client

        # Sampling before min_size_to_sample is reached blocks, then times out.
        try:
            next(client.sample('my_table', num_samples=1, timeout_ms=200,
                               emit_timesteps=False))
        except reverb.errors.DeadlineExceededError:
            print('  Blocked as expected: not enough items yet')

        # Now insert enough items and sample successfully.
        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            for i in range(3):
                writer.append({'v': np.array([i], dtype=np.float32)})
                writer.create_item(table='my_table', priority=1.0, trajectory={'v': writer.history['v'][-1:]})
            writer.flush()
        sample = next(client.sample('my_table', num_samples=1, emit_timesteps=False, timeout_ms=500))
        print(f'  Sampled successfully after inserts: {np.asarray(sample.data[0])}')

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 4: Backpressure with flush(block_until_num_items=N)
# ---------------------------------------------------------------------------
def example_4_backpressure():
    """In production, actors can produce data faster than the server can
    consume it. `flush(block_until_num_items=N)` limits in-flight items to
    prevent OOM. After `flush` returns, at most `N` items are still pending
    (unconfirmed by server).
    """
    print()
    print("=" * 60)
    print("Example 4: Backpressure")
    print("=" * 60)

    server = reverb.Server(
        tables=[reverb.Table.queue(name='q', max_size=100)],
        in_process=True,
    )

    try:
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=10) as writer:
            for i in range(20):
                writer.append({'v': np.array([i], dtype=np.float32)})
                writer.create_item(
                    table='q', priority=1.0,
                    trajectory={'v': writer.history['v'][-1:]},
                )

                # Block if more than 5 items are still in-flight.
                # This prevents the writer from running away from the server.
                writer.flush(block_until_num_items=5, timeout_ms=1000)

            writer.flush()  # final flush: wait for all remaining

        print(f"  Inserted all, table size: {client.server_info()['q'].current_size}")

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 5: append_sequence — batch append
# ---------------------------------------------------------------------------
def example_5_append_sequence():
    """`append_sequence` appends a batch of timesteps in one call. Each element
    in the sequence must share the same leading dimension [T].

    This is equivalent to splitting along the first axis and calling `append`
    once per slice, but avoids stepping through Python per item.
    """
    print()
    print("=" * 60)
    print("Example 5: append_sequence")
    print("=" * 60)

    # NOTE: append_sequence is available on the old `Writer` API (client.writer()),
    # not on `TrajectoryWriter` (client.trajectory_writer()). For new code,
    # prefer TrajectoryWriter and call append() in a loop, unless batch
    # performance is critical.

    server = reverb.Server(
        tables=[reverb.Table.queue(name='q', max_size=100)],
        in_process=True,
    )

    try:
        client = server.in_process_client

        # Use the old `Writer` which supports append_sequence (the C++ method
        # is also available on the gRPC Client's Writer).
        with client.writer(max_sequence_length=10) as writer:
            # Batch of 5 observations, each shape (4,).
            batch_obs = np.array([
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
            ], dtype=np.float32)
            batch_act = np.array([[0], [1], [0], [1], [0]], dtype=np.int64)

            # Append all 5 steps at once.
            writer.append_sequence([batch_obs, batch_act])

            # Create items referencing slices of the buffer.
            writer.create_item('q', num_timesteps=3, priority=1.0)
            writer.create_item('q', num_timesteps=5, priority=1.0)
            writer.flush()

        for sample in client.sample('q', num_samples=2, emit_timesteps=False):
            obs = np.asarray(sample.data[1])  # Columns returned in alphabetical order by key name: 'action' (index 0), 'obs' (index 1).
            print(f"  obs shape: {obs.shape}")

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 6: Writer with delta_encoded / chunk_length (video frames)
# ---------------------------------------------------------------------------
def example_6_delta_encoding():
    """The old `Writer` supports delta encoding: tensors are delta-encoded
    against the first item in their batch before compression. This can
    significantly reduce RAM for highly correlated data (e.g. video frames).

    `chunk_length` controls how many timesteps are grouped together before
    encoding; `max_in_flight_items` limits items awaiting server confirmation.
    """
    print()
    print("=" * 60)
    print("Example 6: Delta Encoding (Writer)")
    print("=" * 60)

    server = reverb.Server(
        tables=[reverb.Table.queue(name='q', max_size=100)],
        in_process=True,
    )

    try:
        client = server.in_process_client

        # Writer with delta encoding enabled.
        with client.writer(
            max_sequence_length=20,
            delta_encoded=True,
            chunk_length=5,
            max_in_flight_items=10,
        ) as writer:
            # Simulate video-like frames: highly correlated.
            base_frame = np.ones((84, 84), dtype=np.uint8) * 100
            for i in range(10):
                frame = base_frame.copy()
                frame[i, i] = 200  # small change
                writer.append([frame])
                writer.create_item('q', num_timesteps=1, priority=1.0)
            writer.flush()

        for sample in client.sample('q', num_samples=3, emit_timesteps=False):
            print(f"  frame shape: {np.asarray(sample.data[0]).shape}")

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 7: client.insert() — direct single-item insert
# ---------------------------------------------------------------------------
def example_7_direct_insert():
    """`client.insert()` inserts a "blob" into one or more tables at once.
    It's a convenience wrapper around `Writer` with `max_sequence_length=1`.
    Prefer `TrajectoryWriter` for streaming inserts.
    """
    print()
    print("=" * 60)
    print("Example 7: client.insert()")
    print("=" * 60)

    server = reverb.Server(
        tables=[reverb.Table.queue(name='q', max_size=100)],
        in_process=True,
    )

    try:
        client = server.in_process_client
        client.insert(
            {'obs': np.zeros(4, dtype=np.float32)},
            priorities={'q': 1.0},
        )
        sample = next(client.sample('q', num_samples=1, emit_timesteps=False))
        print(f"  Sampled: {np.asarray(sample.data[0])}")

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 8: client.reset(table) — clear a table
# ---------------------------------------------------------------------------
def example_8_reset():
    """`client.reset(table)` clears all items and resets the rate limiter.
    Useful for evaluation episodes or experiment restarts.
    """
    print()
    print("=" * 60)
    print("Example 8: client.reset()")
    print("=" * 60)

    server = reverb.Server(
        tables=[reverb.Table.queue(name='q', max_size=100)],
        in_process=True,
    )

    try:
        client = server.in_process_client

        # Insert some data.
        client.insert(np.zeros(4, dtype=np.float32), priorities={'q': 1.0})
        print(f"  Before reset: {client.server_info()['q'].current_size}")

        client.reset('q')
        print(f"  After reset:  {client.server_info()['q'].current_size}")

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 9: Table.can_sample() / Table.can_insert()
# ---------------------------------------------------------------------------
def example_9_rate_limiter_queries():
    """Query whether the rate limiter would allow a sample or insert at the
    current state, without actually performing the operation.
    """
    print()
    print("=" * 60)
    print("Example 9: can_sample / can_insert")
    print("=" * 60)

    # Using a standalone Table for demonstration (no Server needed).
    table = reverb.Table(
        name='test',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=10,
        rate_limiter=reverb.rate_limiters.MinSize(3),
    )
    print(f"  can_sample(1) before inserts: {table.can_sample(1)}")
    print(f"  can_insert(1):               {table.can_insert(1)}")


# ---------------------------------------------------------------------------
# Example 10: Table.replace() — clone with overrides
# ---------------------------------------------------------------------------
def example_10_table_replace():
    """`Table.replace()` creates a new Table with the same configuration,
    overriding only the specified fields. Useful for iterating on table
    settings without reconstructing everything.
    """
    print()
    print("=" * 60)
    print("Example 10: Table.replace()")
    print("=" * 60)

    original = reverb.Table(
        name='original',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(10),
    )

    # Override only the sampler and name.
    cloned = original.replace(
        name='cloned',
        sampler=reverb.selectors.Prioritized(0.8),
    )

    print(f"  Original: name={original.name}, max_size={original.info.max_size}")
    print(f"  Cloned:   name={cloned.name}, max_size={cloned.info.max_size}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    example_1_queue_and_buffer()
    example_2_stack()
    example_3_sample_to_insert_ratio()
    example_4_backpressure()
    example_5_append_sequence()
    example_6_delta_encoding()
    example_7_direct_insert()
    example_8_reset()
    example_9_rate_limiter_queries()
    example_10_table_replace()
    print()
    print("All examples passed.")


if __name__ == '__main__':
    main()
