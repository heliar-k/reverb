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

"""Reverb tutorial — core patterns (numpy, in-process).

This script demonstrates Reverb's embedded / numpy-only mode:
`Server(in_process=True)` holding tables directly in-process, with no
gRPC overhead and no TensorFlow dependency. Data flows as numpy arrays.

The same `Table` / `selectors` / `rate_limiters` / `TrajectoryWriter` APIs
also work over a networked `Server(in_process=False)` with the gRPC `Client`;
see `grpc_client.py` for a minimal round trip.

Run:
    python examples/demo.py
"""

import tempfile

import numpy as np

import reverb

# ---------------------------------------------------------------------------
# Dummy RL environment (pure numpy)
# ---------------------------------------------------------------------------
OBSERVATION_SHAPE = (10, 10)
OBSERVATION_DTYPE = np.uint8
ACTION_SHAPE = (2,)
ACTION_DTYPE = np.float32


def agent_step(unused_timestep) -> np.ndarray:
    return (np.random.uniform(size=ACTION_SHAPE) > 0.5).astype(ACTION_DTYPE)


def environment_step(unused_action) -> np.ndarray:
    return np.random.randint(0, 256, size=OBSERVATION_SHAPE, dtype=OBSERVATION_DTYPE)


# ===========================================================================
# Example 1: Overlapping Trajectories
# ===========================================================================
def example_1_overlapping_trajectories():
    """A TrajectoryWriter keeps a circular buffer of recent data references.
    An item references a slice of that buffer, so successive items can overlap.
    """
    print("=" * 60)
    print("Example 1: Overlapping Trajectories")
    print("=" * 60)

    server = reverb.Server(
        tables=[
            reverb.Table(
                name="my_table",
                sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                remover=reverb.selectors.Fifo(),
                max_size=int(1e6),
                rate_limiter=reverb.rate_limiters.MinSize(2),
            )
        ],
        in_process=True,
    )

    try:
        client = server.in_process_client

        # Dynamically adds trajectories of length 3 to 'my_table'.
        # Circular buffer size: max trajectory length (3-step windows).
        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            timestep = environment_step(None)
            for step in range(4):
                action = agent_step(timestep)
                writer.append({"action": action, "observation": timestep})
                timestep = environment_step(action)

                if step >= 2:
                    # The item consists of the 3 most recent timesteps.
                    writer.create_item(
                        table="my_table",
                        priority=1.5,
                        trajectory={
                            "actions": writer.history["action"][-3:],
                            "observations": writer.history["observation"][-3:],
                        },
                    )

            writer.flush()

        # Sample: emit_timesteps=False returns one ReplaySample per item.
        for sample in client.sample("my_table", num_samples=2, emit_timesteps=False):
            print(
                "  actions shape:", np.asarray(sample.data[0]).shape
            )  # e.g. actions shape: (3, 2)
            print(
                "  observations shape:", np.asarray(sample.data[1]).shape
            )  # e.g. observations shape: (3, 10, 10)

    finally:
        server.stop()


# ===========================================================================
# Example 2: Complete Episodes
# ===========================================================================
def example_2_complete_episodes():
    """Each item is a whole episode.
    The final timestep has an observation but no action; `append` tolerates
    partial steps (missing columns are filled with None).
    """
    print()
    print("=" * 60)
    print("Example 2: Complete Episodes")
    print("=" * 60)

    EPISODE_LENGTH = 150
    NUM_EPISODES = 10

    server = reverb.Server(
        tables=[
            reverb.Table(
                name="my_table",
                sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                remover=reverb.selectors.Fifo(),
                max_size=int(1e6),
                rate_limiter=reverb.rate_limiters.MinSize(2),
            )
        ],
        in_process=True,
    )

    try:
        client = server.in_process_client

        # EPISODE_LENGTH (150) + 1 for the terminal observation.
        with client.trajectory_writer(num_keep_alive_refs=151) as writer:
            for _ in range(NUM_EPISODES):
                timestep = environment_step(None)

                for _ in range(EPISODE_LENGTH):
                    action = agent_step(timestep)
                    writer.append({"action": action, "observation": timestep})
                    timestep = environment_step(action)

                # Terminal observation WITHOUT an action.
                writer.append({"observation": timestep})

                # Action history is one shorter than observation history
                # because the terminal step has no action.
                # Drop the last (None) entry.
                writer.create_item(
                    table="my_table",
                    priority=1.5,
                    trajectory={
                        "actions": writer.history["action"][:-1],
                        "observations": writer.history["observation"][:],
                    },
                )

                # Blocks until insert confirmed, then clears history.
                writer.end_episode(timeout_ms=1000)

                assert len(writer.history["action"]) == 0
                assert len(writer.history["observation"]) == 0

        for sample in client.sample("my_table", num_samples=2, emit_timesteps=False):
            print(
                "  actions shape:", np.asarray(sample.data[0]).shape
            )  # e.g. actions shape: (150, 2)
            print(
                "  observations shape:", np.asarray(sample.data[1]).shape
            )  # e.g. observations shape: (151, 10, 10)

    finally:
        server.stop()


# ===========================================================================
# Example 3: Multiple Priority Tables
# ===========================================================================
def example_3_multiple_tables():
    """One server can hold multiple tables. Items referencing the same data
    elements can live in different tables simultaneously.
    """
    print()
    print("=" * 60)
    print("Example 3: Multiple Priority Tables")
    print("=" * 60)

    server = reverb.Server(
        tables=[
            reverb.Table(
                name="my_table_a",
                sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                remover=reverb.selectors.Fifo(),
                max_size=int(1e6),
                rate_limiter=reverb.rate_limiters.MinSize(2),
            ),
            reverb.Table(
                name="my_table_b",
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=int(1e6),
                rate_limiter=reverb.rate_limiters.MinSize(2),
            ),
        ],
        in_process=True,
    )

    try:
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            timestep = environment_step(None)
            for step in range(4):
                action = agent_step(timestep)
                writer.append({"action": action, "observation": timestep})
                timestep = environment_step(action)

                if step >= 2:
                    # Length-3 trajectory into the prioritized table.
                    writer.create_item(
                        table="my_table_a",
                        priority=1.5,
                        trajectory={
                            "actions": writer.history["action"][-3:],
                            "observations": writer.history["observation"][-3:],
                        },
                    )
                    # Length-2 trajectory into the uniform table.
                    writer.create_item(
                        table="my_table_b",
                        priority=1.0,
                        trajectory={
                            "actions": writer.history["action"][-2:],
                            "observations": writer.history["observation"][-2:],
                        },
                    )

            writer.flush()

        info = client.server_info()
        print("  table_a items:", info["my_table_a"].current_size)
        print("  table_b items:", info["my_table_b"].current_size)

    finally:
        server.stop()


# ===========================================================================
# Example 4: Checkpointing
# ===========================================================================
def example_4_checkpointing():
    """A `Client.checkpoint()` serializes the server's tables to disk.
    A new server with a `DefaultCheckpointer` pointing at the same root
    directory loads the most recent checkpoint on startup.

    The on-disk format is length-delimited protobuf; checkpoints written by
    the old TensorFlow-based Reverb are NOT compatible and must be regenerated.
    """
    print()
    print("=" * 60)
    print("Example 4: Checkpointing")
    print("=" * 60)

    ckpt_root = tempfile.mkdtemp(prefix="reverb_ckpt_")

    def _ckpt_table():
        return reverb.Table(
            name="q",
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=10,
            max_times_sampled=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )

    # Write some data and checkpoint.
    ckpt_server = reverb.Server(
        tables=[_ckpt_table()],
        in_process=True,
        checkpointer=reverb.platform.default.checkpointers.DefaultCheckpointer(
            path=ckpt_root
        ),
    )
    c = ckpt_server.in_process_client
    with c.trajectory_writer(num_keep_alive_refs=1) as w:
        for i in range(5):
            w.append({"v": np.array([i], dtype=np.float32)})
            w.create_item(
                table="q",
                priority=1.0,
                trajectory={"v": w.history["v"][-1:]},
            )
        w.flush()
    path = c.checkpoint()
    print("  checkpoint written to:", path)
    del ckpt_server

    # Restore into a new server.
    restored_server = reverb.Server(
        tables=[_ckpt_table()],
        in_process=True,
        checkpointer=reverb.platform.default.checkpointers.DefaultCheckpointer(
            path=ckpt_root
        ),
    )
    rc = restored_server.in_process_client
    print("  restored table size:", rc.server_info()["q"].current_size)
    for sample in rc.sample("q", num_samples=5, emit_timesteps=False):
        print("  restored value:", np.asarray(sample.data[0]).reshape(-1)[0])
    restored_server.stop()


# ===========================================================================
# Main
# ===========================================================================
def main():
    example_1_overlapping_trajectories()
    example_2_complete_episodes()
    example_3_multiple_tables()
    example_4_checkpointing()
    print()
    print("All examples passed.")


if __name__ == "__main__":
    main()
