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

"""Minimal StructuredWriter example.

`StructuredWriter` derives table insertions from the raw step stream via
*patterns* (which slices of which columns form an item) and *conditions* (when
an item should be emitted). This avoids manually calling `create_item` per
trajectory.

We emit a 2-step window into a FIFO table every 2 steps, and a single most-
recent observation into a second table at end of episode.

Run as a script:

    python examples/structured_writer.py
"""

import numpy as np

import reverb
from reverb import structured_writer as sw


def main() -> None:
  # Two tables: one for 2-step windows, one for last-step snapshots.
  windows = reverb.Table.queue(name='windows', max_size=100)
  last = reverb.Table.queue(name='last', max_size=100)

  server = reverb.Server(tables=[windows, last], in_process=True)
  try:
    client = server.in_process_client

    # Describe the step structure: a dict of columns. Leaves are ignored —
    # only the nesting matters. Each leaf becomes a referenceable column.
    step_structure = {'obs': None, 'reward': None}
    ref = sw.create_reference_step(step_structure)

    # Pattern 1: a 2-step obs window. Emit it every 2 steps.
    window_pattern = {'window': ref['obs'][-2:]}
    window_cfg = sw.create_config(
        pattern=window_pattern,
        table='windows',
        conditions=[sw.Condition.step_index() % 2 == 1],
    )

    # Pattern 2: the most recent obs+reward. Emit at end of episode.
    last_pattern = {'obs': ref['obs'][-1], 'reward': ref['reward'][-1]}
    last_cfg = sw.create_config(
        pattern=last_pattern,
        table='last',
        conditions=[sw.Condition.is_end_episode()],
    )

    writer = client.structured_writer([window_cfg, last_cfg])
    for step in range(5):
      writer.append({
          'obs': np.zeros(4, dtype=np.float32) + step,
          'reward': np.float32(step),
      })
    writer.end_episode()

    # 5 steps, emitting every 2 steps starting at step 1 -> steps 1, 3 => 2 windows.
    window_samples = list(
        client.sample('windows', num_samples=2, emit_timesteps=False))
    assert len(window_samples) == 2
    # End-of-episode fires once => 1 item in 'last'.
    last_samples = list(
        client.sample('last', num_samples=1, emit_timesteps=False))
    assert len(last_samples) == 1
    print(f'windows sampled: {len(window_samples)}, last sampled: {len(last_samples)}')
  finally:
    server.stop()


if __name__ == '__main__':
  main()
