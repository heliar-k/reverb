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

"""Complete RL training loop with PER-style priority updates (numpy, in-process).

Demonstrates:
  - `emit_timesteps=True`  — trajectories returned as per-timestep lists
  - `unpack_as_table_signature=True` — data restored to nested structure
  - Table `signature` (via `signature_codec.TensorSpec`) — schema for unpacking
  - `mutate_priorities` — update item priorities after TD-error recalculation
  - `TrajectoryColumn.numpy()`, `.shape`, `.dtype` — inspecting sampled columns
  - End-to-end: write → sample → compute TD-error → update priorities

This simulates a single-actor, single-learner Prioritized Experience Replay
loop (DQN/DRQN style) with a dummy environment and dummy "network".

Run:
    python examples/training_loop.py
"""

import numpy as np

import reverb
from reverb import signature_codec

# ---------------------------------------------------------------------------
# Dummy RL environment
# ---------------------------------------------------------------------------
OBS_SHAPE = (4,)
ACT_SHAPE = (2,)


class DummyEnv:
    """Gym-like environment producing float32 observations and rewards."""

    def __init__(self):
        self._step = 0
        self._target = 10

    def reset(self):
        self._step = 0
        return np.zeros(OBS_SHAPE, dtype=np.float32)

    def step(self, action: np.ndarray):
        self._step += 1
        obs = np.array([self._step * 0.1] * 4, dtype=np.float32)
        reward = np.float32(0.1 * (action[0] - 1.0))
        done = self._step >= self._target
        return obs, reward, done


# ---------------------------------------------------------------------------
# Dummy "network" and PER helpers
# ---------------------------------------------------------------------------
def compute_q_values(obs: np.ndarray) -> np.ndarray:
    """A dummy Q-network: Q(s, a) = obs[0] + a * 0.1."""
    return np.array([obs[0], obs[0] + 0.1], dtype=np.float32)


def compute_td_error(
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    next_obs: np.ndarray,
    done: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute one-step TD error for DQN-style update."""
    q = compute_q_values(obs)
    q_next = compute_q_values(next_obs)
    # action in the demo is always [0, 1] (one-hot style); the Q-value for
    # the taken action is the second entry (index 1). For a generic action
    # tensor, we'd use argmax but here we simplify.
    action_idx = 1  # demo action is always [0, 1]
    q_val = q[..., action_idx]
    target = reward + gamma * np.max(q_next, axis=-1) * (1 - done)
    return np.abs(target - q_val)


def per_priority(td_error: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """PER priority from TD error (linearised for demo simplicity)."""
    # Real PER: p = (|δ| + ε)^α. We use a linearized version for demo simplicity.
    return td_error + epsilon


# ---------------------------------------------------------------------------
# Table signature for structured sampling
# ---------------------------------------------------------------------------
def build_table_signature():
    """Define the schema of items in the replay table.

    Items are SARS trajectories: [state, action, reward, state'].
    With emit_timesteps=True and unpack_as_table_signature=True, sampled data
    is returned in this exact nested structure.
    """
    return {
        "observation": signature_codec.TensorSpec(
            shape=(None,) + OBS_SHAPE, dtype=np.float32, name="observation"
        ),
        "action": signature_codec.TensorSpec(
            shape=(None,) + ACT_SHAPE, dtype=np.float32, name="action"
        ),
        "reward": signature_codec.TensorSpec(
            shape=(None,), dtype=np.float32, name="reward"
        ),
        "next_observation": signature_codec.TensorSpec(
            shape=(None,) + OBS_SHAPE, dtype=np.float32, name="next_observation"
        ),
        "done": signature_codec.TensorSpec(
            shape=(None,), dtype=np.float32, name="done"
        ),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main():
    # -- Server setup --------------------------------------------------------
    server = reverb.Server(
        tables=[
            reverb.Table(
                name="replay",
                sampler=reverb.selectors.Prioritized(priority_exponent=0.6),
                remover=reverb.selectors.Fifo(),
                max_size=1000,
                rate_limiter=reverb.rate_limiters.MinSize(32),
                signature=build_table_signature(),
            )
        ],
        in_process=True,
    )

    try:
        client = server.in_process_client
        env = DummyEnv()

        # -- Collect a few episodes to fill the buffer -----------------------
        print("Collecting episodes...")
        with client.trajectory_writer(num_keep_alive_refs=50) as writer:
            for ep in range(5):
                obs = env.reset()
                done = False
                while not done:
                    action = np.array([0, 1], dtype=np.float32)
                    next_obs, reward, done = env.step(action)
                    writer.append(
                        {
                            "observation": obs,
                            "action": action,
                            "reward": np.float32(reward),
                            "next_observation": next_obs,
                            "done": np.float32(done),
                        }
                    )

                    # Insert every step as a SARS transition (length-1 trajectory).
                    # For single-step items, use history[-1] for each column.
                    writer.create_item(
                        table="replay",
                        priority=1.0,
                        trajectory={
                            "observation": writer.history["observation"][-1:],
                            "action": writer.history["action"][-1:],
                            "reward": writer.history["reward"][-1:],
                            "next_observation": writer.history["next_observation"][-1:],
                            "done": writer.history["done"][-1:],
                        },
                    )
                    obs = next_obs
                writer.end_episode()
            writer.flush()

        print(f"  Buffer size: {client.server_info()['replay'].current_size}")

        # -- Sample + train + update priorities ------------------------------
        print("Training loop (3 iterations)...")
        for iteration in range(3):
            # Only 3 iterations for demo; in practice run thousands for convergence.
            # Sample with emit_timesteps=True → each item is yielded as a list
            # of per-timestep ReplaySample objects (here length=1 per item).
            # With unpack_as_table_signature=True, each ReplaySample.data is
            # restored to the nested structure defined by the table signature.
            #
            # Note: `emit_timesteps=True` is the DEFAULT for the gRPC Client;
            # here we set it explicitly for clarity.
            samples = list(
                client.sample(
                    "replay",
                    num_samples=8,
                    emit_timesteps=True,  # per-timestep lists
                    unpack_as_table_signature=True,  # nested dict structure
                )
            )

            td_errors = []
            updates = {}

            for timesteps in samples:
                for ts in timesteps:
                    # data is now a dict matching build_table_signature()
                    data = ts.data
                    key = int(ts.info.key)

                    # Extract data as numpy arrays for processing.
                    # (See TrajectoryColumn demo at bottom of file.)
                    obs_arr = np.asarray(data["observation"])
                    act_arr = np.asarray(data["action"])
                    rew_arr = np.asarray(data["reward"])
                    nobs_arr = np.asarray(data["next_observation"])
                    don_arr = np.asarray(data["done"])

                    # Compute TD error.
                    td = compute_td_error(obs_arr, act_arr, rew_arr, nobs_arr, don_arr)

                    # Store for priority update.
                    updates[key] = float(per_priority(td))
                    td_errors.append(float(td))

            # Update priorities for all sampled items.
            client.mutate_priorities("replay", updates=updates)

            print(
                f"  iteration {iteration + 1}: "
                f"mean TD error = {np.mean(td_errors):.4f}, "
                f"updated {len(updates)} items"
            )

        # -- TrajectoryColumn API demo on a freshly inserted item ------------
        print()
        print("TrajectoryColumn API demo:")
        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            for i in range(3):
                writer.append(
                    {
                        "observation": np.ones(OBS_SHAPE, dtype=np.float32) * i,
                        "action": np.zeros(ACT_SHAPE, dtype=np.float32),
                        "reward": np.float32(i),
                        "next_observation": np.ones(OBS_SHAPE, dtype=np.float32)
                        * (i + 1),
                        "done": np.float32(0),
                    }
                )

            # Access a TrajectoryColumn before creating an item.
            col = writer.history["observation"][:]
            print(f"  observation column shape: {col.shape}")
            print(f"  observation column dtype:  {col.dtype}")
            arr = col.numpy()
            print(f"  observation numpy() shape: {arr.shape}")
            print(f"  observation numpy():\n{arr}")

            writer.create_item(
                table="replay",
                priority=1.0,
                trajectory={
                    "observation": writer.history["observation"][:],
                    "action": writer.history["action"][:],
                    "reward": writer.history["reward"][:],
                    "next_observation": writer.history["next_observation"][:],
                    "done": writer.history["done"][:],
                },
            )
            writer.flush()

        # Sample back to verify.
        for sample in client.sample(
            "replay",
            num_samples=1,
            emit_timesteps=False,
            unpack_as_table_signature=True,
        ):
            data = sample.data
            print(
                f"\n  Sampled back: obs={np.asarray(data['observation']).shape}, "
                f"act={np.asarray(data['action']).shape}"
            )

    finally:
        server.stop()

    print("\nDone.")


if __name__ == "__main__":
    main()
