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

"""Table signatures: schema definition and structured unpacking (numpy, in-process).

A table `signature` describes the nested structure and dtypes of items.
When sampling with `unpack_as_table_signature=True`, data is returned in
that exact structure — no manual unflattening needed.

Demonstrates:
  - `signature_codec.TensorSpec` — dtype/shape spec for each column
  - `Table(signature=...)` — attach a schema to a table
  - `unpack_as_table_signature=True` — auto-unflat on sample
  - Nested signatures (dict-of-dict, list, tuple)
  - Signature validation on insert (rejects mismatched dtypes)

Run:
    python examples/table_signature.py
"""

import numpy as np

import reverb
from reverb import signature_codec


# ---------------------------------------------------------------------------
# Example 1: Flat signature
# ---------------------------------------------------------------------------
def example_1_flat_signature():
    """Basic table signature: each key maps to a TensorSpec."""
    print("=" * 60)
    print("Example 1: Flat signature")
    print("=" * 60)

    signature = {
        "observation": signature_codec.TensorSpec(
            shape=(None, 4), dtype=np.float32, name="observation"
        ),
        "action": signature_codec.TensorSpec(
            shape=(None, 2), dtype=np.float32, name="action"
        ),
    }

    # shape=(None, 4): None = variable time dimension (trajectory length), 4 = feature dim.
    # `name` is a debug/display label; it does not affect serialization or sampling.

    table = reverb.Table(
        name="replay",
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=signature,
    )

    server = reverb.Server(tables=[table], in_process=True)
    try:
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=5) as writer:
            for i in range(3):
                writer.append(
                    {
                        "observation": np.ones((4,), dtype=np.float32) * i,
                        "action": np.zeros((2,), dtype=np.float32),
                    }
                )
            writer.create_item(
                table="replay",
                priority=1.0,
                trajectory={
                    "observation": writer.history["observation"][:],
                    "action": writer.history["action"][:],
                },
            )
            writer.flush()

        # Without unpack_as_table_signature: data is a flat list.
        sample_flat = next(client.sample("replay", num_samples=1, emit_timesteps=False))
        print(f"  Flat data type: {type(sample_flat.data)}")
        print(f"  Flat data[0] shape: {np.asarray(sample_flat.data[0]).shape}")

        # With unpack_as_table_signature: data is the original dict.
        sample_unpacked = next(
            client.sample(
                "replay",
                num_samples=1,
                emit_timesteps=False,
                unpack_as_table_signature=True,
            )
        )
        print(f"  Unpacked data type: {type(sample_unpacked.data)}")
        print(f"  Unpacked keys: {list(sample_unpacked.data.keys())}")
        print(
            f"  observation shape: "
            f"{np.asarray(sample_unpacked.data['observation']).shape}"
        )

        # --- Signature validation (uncomment to see dtype mismatch error) ---
        # with client.trajectory_writer(num_keep_alive_refs=1) as writer:
        #     writer.append({'observation': np.ones(4, dtype=np.float64)})  # float64 != float32
        #     writer.create_item(table='replay', priority=1.0,
        #                        trajectory={'observation': writer.history['observation'][:]})
        #     writer.flush()
        # ValueError: dtype mismatch for column observation

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 2: Nested signature (dict-of-dict)
# ---------------------------------------------------------------------------
def example_2_nested_signature():
    """Signatures can be arbitrarily nested dicts. This is useful when the
    step data has a hierarchical structure (e.g. sensor readings grouped
    by modality).
    """
    print()
    print("=" * 60)
    print("Example 2: Nested signature (dict-of-dict)")
    print("=" * 60)

    signature = {
        "sensors": {
            "camera": signature_codec.TensorSpec(
                shape=(None, 64, 64, 3), dtype=np.uint8, name="camera"
            ),
            "lidar": signature_codec.TensorSpec(
                shape=(None, 16), dtype=np.float32, name="lidar"
            ),
        },
        "action": signature_codec.TensorSpec(
            shape=(None, 2), dtype=np.float32, name="action"
        ),
    }

    table = reverb.Table(
        name="replay",
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=signature,
    )

    server = reverb.Server(tables=[table], in_process=True)
    try:
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            for i in range(2):
                writer.append(
                    {
                        "sensors": {
                            "camera": np.ones((64, 64, 3), dtype=np.uint8) * (i + 1),
                            "lidar": np.zeros(16, dtype=np.float32) + i,
                        },
                        "action": np.array([i, i + 1], dtype=np.float32),
                    }
                )
            writer.create_item(
                table="replay",
                priority=1.0,
                trajectory={
                    "sensors": {
                        "camera": writer.history["sensors"]["camera"][:],
                        "lidar": writer.history["sensors"]["lidar"][:],
                    },
                    "action": writer.history["action"][:],
                },
            )
            writer.flush()

        sample = next(
            client.sample(
                "replay",
                num_samples=1,
                emit_timesteps=False,
                unpack_as_table_signature=True,
            )
        )
        data = sample.data
        # Access nested fields.
        camera = np.asarray(data["sensors"]["camera"])
        lidar = np.asarray(data["sensors"]["lidar"])
        action = np.asarray(data["action"])
        print(f"  camera shape: {camera.shape}")
        print(f"  lidar shape:  {lidar.shape}")
        print(f"  action shape: {action.shape}")

    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Example 3: List and Tuple signatures
# ---------------------------------------------------------------------------
def example_3_list_tuple_signature():
    """Signatures can also be lists or tuples (not just dicts)."""
    print()
    print("=" * 60)
    print("Example 3: List and Tuple signatures")
    print("=" * 60)

    # Tuple signature.
    signature = (
        signature_codec.TensorSpec(shape=(None, 3), dtype=np.float32, name="obs"),
        signature_codec.TensorSpec(shape=(None, 1), dtype=np.int64, name="act"),
        signature_codec.TensorSpec(shape=(None,), dtype=np.float32, name="rew"),
    )

    table = reverb.Table(
        name="replay",
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=signature,
    )

    server = reverb.Server(tables=[table], in_process=True)
    try:
        client = server.in_process_client

        with client.trajectory_writer(num_keep_alive_refs=3) as writer:
            for i in range(3):
                writer.append(
                    (
                        np.ones(3, dtype=np.float32) * i,
                        np.array([i], dtype=np.int64),
                        np.float32(i * 0.1),
                    )
                )
            # Tuple signatures use positional indexing: writer.history[i] for the i-th element.
            writer.create_item(
                table="replay",
                priority=1.0,
                trajectory=(
                    writer.history[0][:],
                    writer.history[1][:],
                    writer.history[2][:],
                ),
            )
            writer.flush()

        sample = next(
            client.sample(
                "replay",
                num_samples=1,
                emit_timesteps=False,
                unpack_as_table_signature=True,
            )
        )
        obs, act, rew = [np.asarray(x) for x in sample.data]
        print(f"  obs: {obs.shape}, act: {act.shape}, rew: {rew.shape}")

    finally:
        server.stop()


# ===========================================================================
# Main
# ===========================================================================
def main():
    example_1_flat_signature()
    example_2_nested_signature()
    example_3_list_tuple_signature()
    print()
    print("All examples passed.")


if __name__ == "__main__":
    main()
