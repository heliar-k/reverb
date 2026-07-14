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

r"""Advanced StructuredWriter patterns (numpy, in-process).

`StructuredWriter` derives table insertions from the raw step stream via
*patterns* (which slices of which columns form an item) and *conditions*
(when an item should be emitted). This avoids manually calling `create_item`
per trajectory.

Demonstrates:
  - `td_error()` priority function — automatic PER-style priority
  - `Condition.steps_since_applied()` — throttle insertions by recency
  - `Condition.data()` — condition on scalar fields in the step data
  - Modulo conditions — periodic insertions (every-N-steps)
  - `pattern_from_transform()` — functional pattern construction
  - `partial_step=True` — on-policy: append observation, sample, learn, then
    append action in the same logical step
  - `infer_signature()` — derive table schema from StructuredWriter configs

Run:
    python examples/structured_writer_advanced.py
"""

from typing import Any

import numpy as np
import tree

import reverb
from reverb import signature_codec
from reverb import structured_writer as sw


# ---------------------------------------------------------------------------
# Dummy "environment" and "network"
# ---------------------------------------------------------------------------
def make_step(step_id: int) -> dict:
    """A single RL step with observation, action, reward, and a per-step
    TD error used for priority calculation."""
    return {
        'obs': np.zeros(4, dtype=np.float32) + step_id,
        'action': np.array([step_id % 2], dtype=np.int64),
        'reward': np.float32(0.1 * step_id),
        'td_error': np.float32(0.5 - step_id * 0.05),  # mock TD error
        'done': np.bool_(step_id >= 8),
    }


STEP_STRUCTURE = tree.map_structure(lambda _: None, make_step(0))


# ===========================================================================
# Example 1: td_error() priority
# ===========================================================================
def example_1_td_error_priority():
    r"""`td_error()` computes priority from a TD-error column within the step.

    For a trajectory of length T, the priority is:
        max_priority_weight * max(|td_error|) +
        (1 - max_priority_weight) * mean(|td_error|)

    This mirrors the R2D2 paper's priority scheme.
    """
    print("=" * 60)
    print("Example 1: td_error() priority")
    print("=" * 60)

    table = reverb.Table.queue(name='replay', max_size=100)
    server = reverb.Server(tables=[table], in_process=True)

    try:
        client = server.in_process_client

        ref = sw.create_reference_step(STEP_STRUCTURE)

        # Pattern: last 3 obs + action, with a td_error field for priority.
        pattern = {
            'obs': ref['obs'][-3:],
            'action': ref['action'][-3:],
        }

        # Priority = f(td_error over the trajectory).
        # `step_structure` tells td_error() how to traverse the step tree.
        # `get_field_from_step_fn` picks the field containing per-step TD errors.
        # The priority of the trajectory is: 0.9 * max(|td_error|) + 0.1 * mean(|td_error|).
        priority_fn = sw.td_error(
            max_priority_weight=0.9,
            step_structure=STEP_STRUCTURE,
            get_field_from_step_fn=lambda s: s['td_error'],
        )

        config = sw.create_config(
            pattern=pattern,
            table='replay',
            conditions=[sw.Condition.step_index() >= 2],
            priority=priority_fn,
        )

        writer = client.structured_writer([config])
        for step_id in range(10):
            writer.append(make_step(step_id))

        # Items were inserted with td_error-based priorities.
        info = client.server_info()
        print(f"  Table size: {info['replay'].current_size}")
        for sample in client.sample('replay', num_samples=2,
                                    emit_timesteps=False):
            print(f"  Priority: {sample.info.priority:.4f}, "
                  f"obs mean: {np.asarray(sample.data[0]).mean():.1f}")

    finally:
        server.stop()


# ===========================================================================
# Example 2: steps_since_applied() throttle
# ===========================================================================
def example_2_steps_since_applied():
    """`Condition.steps_since_applied()` is the number of steps since this
    config last emitted an item. Use it to enforce a cooldown: e.g. emit
    at most every 3 steps.

    Combine with other conditions: e.g. emit when step_index is a multiple
    of 2 AND at least 3 steps have passed since the last emission.
    """
    print()
    print("=" * 60)
    print("Example 2: steps_since_applied() throttle")
    print("=" * 60)

    table = reverb.Table.queue(name='replay', max_size=100)
    server = reverb.Server(tables=[table], in_process=True)

    try:
        client = server.in_process_client
        ref = sw.create_reference_step(STEP_STRUCTURE)

        config = sw.create_config(
            pattern={'obs': ref['obs'][-2:]},
            table='replay',
            conditions=[
                # Emit when step index is odd AND at least 3 steps since last.
                # Multiple conditions are AND-ed: both must be true for an item to be emitted.
                sw.Condition.step_index() % 2 == 1,
                sw.Condition.steps_since_applied() >= 3,
            ],
        )

        writer = client.structured_writer([config])
        for step_id in range(12):
            writer.append(make_step(step_id))

        # 12 steps, emit when step_index in {1,3,5,7,9,11} (odd)
        # constrained by steps_since_applied >= 3:
        #   step 1: first emission (no previous)
        #   step 3: 2 steps since, throttled
        #   step 5: 4 steps since → emit
        #   step 7: 2 steps since, throttled
        #   step 9: 4 steps since → emit
        #   step 11: 2 steps since, throttled
        # Total: 3 items.
        info = client.server_info()
        print(f"  Table size: {info['replay'].current_size} (expected 3)")

    finally:
        server.stop()


# ===========================================================================
# Example 3: Condition.data() — condition on step fields
# ===========================================================================
def example_3_condition_on_data():
    """`Condition.data()` builds conditions on scalar fields in the step data.
    For example: only emit when `done` is True (episode boundary).
    """
    print()
    print("=" * 60)
    print("Example 3: Condition.data()")
    print("=" * 60)

    table = reverb.Table.queue(name='replay', max_size=100)
    server = reverb.Server(tables=[table], in_process=True)

    try:
        client = server.in_process_client
        ref = sw.create_reference_step(STEP_STRUCTURE)

        # Build a condition on the 'done' field: emit when done == 1.
        done_condition = sw.Condition.data(STEP_STRUCTURE)['done'] == 1

        config = sw.create_config(
            pattern={'obs': ref['obs'][-3:]},
            table='replay',
            conditions=[done_condition],
        )

        writer = client.structured_writer([config])
        for step_id in range(10):
            writer.append(make_step(step_id))

        # 'done' is 1.0 only at step_id >= 8. Items are 3-step windows,
        # so we get insertions at steps 8 and 9 (and possibly step 10? No,
        # only 10 steps total, step 9 is the last). Step 8: done=true,
        # pattern is [-3:] of obs = steps 6,7,8. Step 9: done=true,
        # pattern = steps 7,8,9.
        info = client.server_info()
        print(f"  Table size: {info['replay'].current_size} (expected 2)")

    finally:
        server.stop()


# ===========================================================================
# Example 4: Modulo conditions — periodic insertions
# ===========================================================================
def example_4_modulo_conditions():
    """Using `step_index() % N` for periodic insertions. The modulo operator
    on conditions is syntactic sugar that sets `mod_eq.mod`; equality then
    sets `mod_eq.eq`.
    """
    print()
    print("=" * 60)
    print("Example 4: Modulo conditions")
    print("=" * 60)

    table = reverb.Table.queue(name='replay', max_size=100)
    server = reverb.Server(tables=[table], in_process=True)

    try:
        client = server.in_process_client
        ref = sw.create_reference_step(STEP_STRUCTURE)

        # Emit every 3rd step, starting from index 2.
        config = sw.create_config(
            pattern={'obs': ref['obs'][-2:]},      # 2-step window
            table='replay',
            conditions=[sw.Condition.step_index() % 3 == 2],
        )

        writer = client.structured_writer([config])
        for step_id in range(10):
            writer.append(make_step(step_id))

        # Indices: 2, 5, 8 → 3 insertions.
        info = client.server_info()
        print(f"  Table size: {info['replay'].current_size} (expected 3)")

    finally:
        server.stop()


# ===========================================================================
# Example 5: pattern_from_transform() — functional pattern
# ===========================================================================
def example_5_pattern_from_transform():
    """`pattern_from_transform()` builds a pattern from a function that maps
    a reference step to a trajectory structure. This is often more readable
    than building patterns node-by-node.
    """
    print()
    print("=" * 60)
    print("Example 5: pattern_from_transform()")
    print("=" * 60)

    table = reverb.Table.queue(name='replay', max_size=100)
    server = reverb.Server(tables=[table], in_process=True)

    try:
        client = server.in_process_client

        def my_transform(step: sw.ReferenceStep) -> Any:
            """Build a SARS trajectory: last 3 states, 2 actions, 2 rewards."""
            return {
                'states': step['obs'][-3:],
                'actions': step['action'][-2:],
                'rewards': step['reward'][-2:],
            }

        pattern = sw.pattern_from_transform(STEP_STRUCTURE, my_transform)

        config = sw.create_config(
            pattern=pattern,
            table='replay',
            conditions=[sw.Condition.step_index() >= 2],
        )

        writer = client.structured_writer([config])
        for step_id in range(8):
            writer.append(make_step(step_id))

        info = client.server_info()
        print(f"  Table size: {info['replay'].current_size}")

        for sample in client.sample('replay', num_samples=1,
                                    emit_timesteps=False):
            states = np.asarray(sample.data[2])
            actions = np.asarray(sample.data[0])
            rewards = np.asarray(sample.data[1])
            print(f"  states: {states.shape}, actions: {actions.shape}, "
                  f"rewards: {rewards.shape}")

    finally:
        server.stop()


# ===========================================================================
# Example 6: partial_step — on-policy agent
# ===========================================================================
def example_6_partial_step():
    """On-policy agents need to sample & learn from SARS before picking the
    next action. `partial_step=True` lets you append observations first,
    trigger insertions (which can be sampled), then append the chosen action
    to complete the step — all within a single logical step index.

    The writer holds the step open until `append(partial_step=False)`.
    """
    print()
    print("=" * 60)
    print("Example 6: partial_step (on-policy)")
    print("=" * 60)

    table = reverb.Table(
        name='replay',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )
    server = reverb.Server(tables=[table], in_process=True)

    try:
        client = server.in_process_client
        ref = sw.create_reference_step(STEP_STRUCTURE)

        # Pattern: a 2-step SARS trajectory (obs + action + reward).
        config = sw.create_config(
            pattern={
                'obs': ref['obs'][-2:],
                'action': ref['action'][-2:],
                'reward': ref['reward'][-2:],
            },
            table='replay',
            conditions=[sw.Condition.step_index() >= 1],
        )

        writer = client.structured_writer([config])

        # Step 0: append observation only (partial), "learn" to pick action.
        # StructuredWriter requires ALL fields in every append call, even
        # partial ones. Missing fields must be passed as None.
        step0 = make_step(0)
        writer.append({
            'obs': step0['obs'],
            'action': None,
            'reward': None,
            'td_error': None,
            'done': None,
        }, partial_step=True)
        # In a real agent: sample from replay, compute policy, pick action.
        # Here we just close the step with the remaining fields.
        writer.append({
            'obs': None,
            'action': step0['action'],
            'reward': step0['reward'],
            'td_error': step0['td_error'],
            'done': step0['done'],
        }, partial_step=False)

        # Show the on-policy flow: append obs → sample & learn → append action.
        for step_id in range(1, 6):
            step = make_step(step_id)
            writer.append({
                'obs': step['obs'],
                'action': None,
                'reward': None,
                'td_error': None,
                'done': None,
            }, partial_step=True)
            # In a real agent: items must be flushed before they can be sampled.
            writer.flush()
            # Sample trajectories from replay to compute the policy.
            try:
                for sample in client.sample('replay', num_samples=1,
                                            emit_timesteps=False, timeout_ms=500):
                    print(f'  Sampled during partial step {step_id}: '
                          f'obs shape={np.asarray(sample.data[0]).shape}')
            except reverb.errors.DeadlineExceededError:
                print(f'  Partial step {step_id}: no items available yet '
                      f'(rate limiter blocked)')
            writer.append({
                'obs': None,
                'action': step['action'],
                'reward': step['reward'],
                'td_error': step['td_error'],
                'done': step['done'],
            }, partial_step=False)

        writer.flush()

        info = client.server_info()
        print(f"  Table size: {info['replay'].current_size}")

    finally:
        server.stop()


# ===========================================================================
# Example 7: infer_signature() — derive table schema
# ===========================================================================
def example_7_infer_signature():
    """`infer_signature()` derives the table's item schema from the
    StructuredWriter configs that target it. This yields a nested structure
    of `TensorSpec` that can be used as the table's `signature` argument.

    This avoids manually defining the signature twice.
    """
    print()
    print("=" * 60)
    print("Example 7: infer_signature()")
    print("=" * 60)

    configs = []
    ref = sw.create_reference_step(STEP_STRUCTURE)

    # Config 1: 3-step obs windows (emitted every step after index 2).
    configs.append(sw.create_config(
        pattern={'observation': ref['obs'][-3:]},
        table='replay',
        conditions=[sw.Condition.step_index() >= 2],
    ))

    # Define the step spec: what each `writer.append()` call provides.
    # Each leaf is a numpy array; infer_signature reads .dtype and .shape.
    step_spec = tree.map_structure(
        lambda x: x,  # actual arrays from a real step
        make_step(0),
    )

    inferred = sw.infer_signature(configs, step_spec)
    print("  Inferred signature:")
    for path, spec in tree.flatten_with_path(inferred):
        print(f"    {path}: shape={spec.shape}, dtype={spec.dtype}")

    # Use the inferred signature to construct the table.
    table = reverb.Table(
        name='replay',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=inferred,
    )

    server = reverb.Server(tables=[table], in_process=True)
    try:
        client = server.in_process_client
        writer = client.structured_writer(configs)
        for step_id in range(6):
            writer.append(make_step(step_id))
        writer.flush()

        # Sample with unpack_as_table_signature.
        for sample in client.sample(
            'replay', num_samples=2, emit_timesteps=False,
            unpack_as_table_signature=True,
        ):
            data = sample.data
            print(f"  Unpacked: obs shape={np.asarray(data['observation']).shape}")

    finally:
        server.stop()


# ===========================================================================
# Main
# ===========================================================================
def main():
    example_1_td_error_priority()
    example_2_steps_since_applied()
    example_3_condition_on_data()
    example_4_modulo_conditions()
    example_5_pattern_from_transform()
    example_6_partial_step()
    example_7_infer_signature()
    print()
    print("All examples passed.")


if __name__ == '__main__':
    main()
