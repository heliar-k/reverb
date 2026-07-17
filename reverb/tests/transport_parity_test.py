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

"""Three-transport parity test: gRPC Client vs LocalClient vs ShmClient.

`Client`, `LocalClient`, and `ShmClient` all inherit `_BaseClient` and should
agree on the shared `trajectory_writer`/`sample` surface. The gRPC vs LocalClient
pair is already guarded by `in_process_test.ClientLocalClientParityTest`; this
file extends the guard to include SHM, so a regression in any one transport's
write/read semantics surfaces as a cross-transport mismatch.

SHM supports all tables (routed by table name, ticket ⑨); each test builds a
single-table server per transport. `mutate_priorities`/`reset` are SHM-
supported as of ticket ⑩ (riding the insert flow under a client mutex);
`server_info` is a bootstrap-time snapshot on SHM (does not reflect
mid-session `Table.replace`), so live-`server_info` parity is asserted only
between gRPC and LocalClient. The shared write→sample round trip remains the
primary parity guard.
"""

import numpy as np
from absl.testing import absltest, parameterized

import reverb

TABLE = "t"


def _make_table(max_size=100, max_times_sampled=1):
    return reverb.Table(
        name=TABLE,
        sampler=reverb.selectors.Fifo(),
        remover=reverb.selectors.Fifo(),
        max_size=max_size,
        max_times_sampled=max_times_sampled,
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )


def _make_clients():
    """Returns (clients, servers): one client per transport, each backed by its
    own single-table server."""
    # gRPC: networked server + Client.
    grpc_server = reverb.Server(tables=[_make_table()], in_process=False)
    grpc_client = reverb.Client(f"localhost:{grpc_server.port}")
    # LocalClient: in-process server.
    local_server = reverb.Server(tables=[_make_table()], in_process=True)
    local_client = local_server.in_process_client
    # ShmClient: in-process server + SHM transport layered on the same table.
    shm_server = reverb.Server(tables=[_make_table()], in_process=True, shm=True)
    shm_client = reverb.ShmClient(shm_server.shm_socket_path)
    return (
        {"grpc": grpc_client, "local": local_client, "shm": shm_client},
        {"grpc": grpc_server, "local": local_server, "shm": shm_server},
    )


def _write_trajectory(client, obs_values):
    """Writes a single multi-step trajectory item; mirrors examples/grpc_client.py."""
    with client.trajectory_writer(num_keep_alive_refs=len(obs_values)) as writer:
        for v in obs_values:
            writer.append({"obs": np.asarray(v, dtype=np.float32)})
        writer.create_item(
            table=TABLE,
            priority=1.0,
            trajectory={"obs": writer.history["obs"][:]},
        )
        writer.flush()


def _sample_obs(client, num_samples=1):
    """Samples and returns the stacked obs array (emit_timesteps=False)."""
    samples = list(client.sample(TABLE, num_samples=num_samples, emit_timesteps=False))
    assert len(samples) == num_samples, f"got {len(samples)} samples"
    return np.asarray(samples[0].data[0])


class ThreeTransportParityTest(parameterized.TestCase):
    """Asserts gRPC / LocalClient / ShmClient agree on trajectory write→sample."""

    def test_single_step_trajectory_parity(self):
        clients, servers = _make_clients()
        try:
            for c in clients.values():
                _write_trajectory(c, [42.0])
            results = {name: _sample_obs(c) for name, c in clients.items()}
            # All three must return the same single-step obs.
            self.assertEqual(results["grpc"].shape, (1,))
            np.testing.assert_array_equal(results["grpc"], [42.0])
            np.testing.assert_array_equal(results["local"], results["grpc"])
            np.testing.assert_array_equal(results["shm"], results["grpc"])
        finally:
            for s in servers.values():
                s.stop()

    @parameterized.named_parameters(
        ("three_steps", [0.0, 1.0, 2.0]),
        ("five_steps", [10.0, 20.0, 30.0, 40.0, 50.0]),
    )
    def test_multi_step_trajectory_parity(self, obs_values):
        clients, servers = _make_clients()
        try:
            for c in clients.values():
                _write_trajectory(c, obs_values)
            results = {name: _sample_obs(c) for name, c in clients.items()}
            expected = np.asarray(obs_values, dtype=np.float32)
            self.assertEqual(results["grpc"].shape, expected.shape)
            np.testing.assert_array_equal(results["grpc"], expected)
            np.testing.assert_array_equal(results["local"], results["grpc"])
            np.testing.assert_array_equal(results["shm"], results["grpc"])
        finally:
            for s in servers.values():
                s.stop()

    def test_multi_column_trajectory_parity(self):
        # Two columns (obs + action): data[0]=action (alpha-first), data[1]=obs.
        clients, servers = _make_clients()
        try:
            for c in clients.values():
                with c.trajectory_writer(num_keep_alive_refs=3) as writer:
                    for step in range(3):
                        writer.append(
                            {
                                "obs": np.zeros(4, dtype=np.float32) + step,
                                "action": np.array([step], dtype=np.int64),
                            }
                        )
                    writer.create_item(
                        table=TABLE,
                        priority=1.0,
                        trajectory={
                            "obs": writer.history["obs"][:],
                            "action": writer.history["action"][:],
                        },
                    )
                    writer.flush()
            results = {}
            for name, c in clients.items():
                sample = list(c.sample(TABLE, num_samples=1, emit_timesteps=False))[0]
                # data columns in tree-flatten (alphabetical key) order: action, obs.
                results[name] = (np.asarray(sample.data[0]), np.asarray(sample.data[1]))
            # obs column shape (3,4), action column shape (3,1) on all transports.
            for name, (action, obs) in results.items():
                self.assertEqual(action.shape, (3, 1), f"{name} action shape")
                self.assertEqual(obs.shape, (3, 4), f"{name} obs shape")
            np.testing.assert_array_equal(results["local"][1], results["grpc"][1])
            np.testing.assert_array_equal(results["shm"][1], results["grpc"][1])
        finally:
            for s in servers.values():
                s.stop()

    def test_mutate_and_reset_parity(self):
        # ticket ⑩: mutate_priorities + reset now agree across all three
        # transports. Insert one item, run mutate_priorities (absent key: no-op
        # per MutateItems semantics) + reset, then confirm a fresh insert still
        # samples back (mirrors client_test.py::test_reset: reset clears the
        # table but leaves it usable). We do NOT sample an empty table after
        # reset because gRPC's NewSampler silently ignores timeout_ms and would
        # block forever on MinSize(1); instead we verify reset by re-inserting
        # and sampling the new item.
        clients, servers = _make_clients()
        try:
            for c in clients.values():
                _write_trajectory(c, [7.0])
            for c in clients.values():
                c.mutate_priorities(TABLE, updates={999: 5.0})  # absent key: no-op
                c.reset(TABLE)
            # After reset the table is empty but usable: re-insert and sample.
            for c in clients.values():
                _write_trajectory(c, [9.0])
            for name, c in clients.items():
                obs = _sample_obs(c)
                np.testing.assert_array_equal(obs, [9.0], err_msg=name)
        finally:
            for s in servers.values():
                s.stop()


if __name__ == "__main__":
    absltest.main()
