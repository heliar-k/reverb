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

"""Minimal shared-memory (SHM) client example.

Same-machine cross-process transport: zero-copy, no serialization, ~9-11x
faster than gRPC loopback (see docs/shm-benchmark.md). `ShmClient` mirrors
`Client`/`LocalClient`, so only the transport differs.

v1: `server_info` returns a connect-time bootstrap snapshot (real `TableInfo`,
but does not reflect mid-session `Table.replace` — use gRPC/Local for live
signatures). `mutate_priorities`/`reset`/`checkpoint` are supported over SHM
(tickets ⑩/⑪, riding the insert flow). Legacy `writer`/`insert` raise
`NotImplementedError` — use `trajectory_writer`/`structured_writer`.

Run as a script:

    python examples/shm_client.py
"""

import numpy as np

import reverb


def main() -> None:
    table = reverb.Table(
        name="experience",
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )

    # in_process=True owns the table directly; shm=True layers the SHM transport
    # on top of it. shm=True does NOT imply in_process=True.
    server = reverb.Server(tables=[table], in_process=True, shm=True)
    try:
        client = reverb.ShmClient(server.shm_socket_path)

        # Identical API to LocalClient / gRPC Client.
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

        samples = list(client.sample("experience", num_samples=1, emit_timesteps=False))
        assert len(samples) == 1
        # data columns are in tree-flatten (alphabetical key) order: action, obs.
        action = np.asarray(samples[0].data[0])
        obs = np.asarray(samples[0].data[1])
        assert obs.shape == (3, 4), obs.shape
        assert action.shape == (3, 1), action.shape
        print(f"sampled obs:\n{obs}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
