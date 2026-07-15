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

"""Minimal gRPC client/server example (no TensorFlow).

A networked `Server(in_process=False)` + `Client('localhost:port')` round trip:
build a table, write one trajectory, sample it back. Data flows as numpy arrays.

Run as a script:

    python examples/grpc_client.py
"""

import numpy as np

import reverb


def main() -> None:
    # A FIFO replay table of capacity 100; samples are allowed once >=1 item.
    table = reverb.Table(
        name="experience",
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )

    # Networked server: clients connect over gRPC to server.port.
    server = reverb.Server(tables=[table], in_process=False)
    try:
        client = reverb.Client(f"localhost:{server.port}")

        # Stream a 3-step trajectory, then create one item referencing all of it.
        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            for step in range(3):
                writer.append(
                    {
                        "obs": np.zeros(4, dtype=np.float32) + step,
                        "action": np.array([step], dtype=np.int64),
                    }
                )
            writer.create_item(
                table="experience",
                priority=1.0,
                trajectory={
                    "obs": writer.history["obs"][:],
                    "action": writer.history["action"][:],
                },
            )
            writer.flush()

        # Sample it back: one ReplaySample per item (emit_timesteps=False).
        # data columns are in tree-flatten (alphabetical key) order: action, obs.
        samples = list(client.sample("experience", num_samples=1, emit_timesteps=False))
        assert len(samples) == 1
        action = np.asarray(samples[0].data[0])
        obs = np.asarray(samples[0].data[1])
        assert obs.shape == (3, 4), obs.shape
        assert action.shape == (3, 1), action.shape
        print(f"sampled obs:\n{obs}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
