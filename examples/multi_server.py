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

"""Multi-server horizontal scaling (numpy, gRPC).

Reverb servers are unaware of each other. Horizontal scaling is achieved by
running multiple Server instances and distributing clients across them. This
example demonstrates:

  - Running multiple servers on different ports
  - Connecting separate clients to each server
  - Round-robin load balancing across servers
  - `Server.wait()` — blocking until shutdown
  - `Server.localhost_client()` — create a gRPC client to localhost

For production, use a gRPC-compatible load balancer in front of the servers
and point a single `Client` to the load-balanced address.

Run:
    python examples/multi_server.py
"""

import threading
import time

import numpy as np

import reverb


# ---------------------------------------------------------------------------
# Example 1: Multiple servers with round-robin clients
# ---------------------------------------------------------------------------
def example_1_round_robin():
    """Run N servers, insert data round-robin, sample from each."""
    print("=" * 60)
    print("Example 1: Round-robin across servers")
    print("=" * 60)

    NUM_SERVERS = 3
    servers: list[reverb.Server] = []
    clients: list[reverb.Client] = []

    for i in range(NUM_SERVERS):
        server = reverb.Server(
            tables=[
                reverb.Table.queue(name="experience", max_size=100),
            ],
            in_process=False,  # gRPC mode
        )
        servers.append(server)
        clients.append(reverb.Client(f"localhost:{server.port}"))
        print(f"  Server {i} on port {server.port}")

    try:
        # ⚠️  DEMONSTRATION ONLY: manual round-robin. In production, run servers
        # behind a gRPC-compatible load balancer (e.g. Envoy, nginx, Traefik)
        # and connect a single Client to the load-balanced address. The load
        # balancer distributes operations across nodes automatically.

        # Create writers. Using explicit __enter__/__exit__ instead of a list
        # comprehension context manager for cleaner error handling.
        writers = [
            client.trajectory_writer(num_keep_alive_refs=3) for client in clients
        ]
        for w in writers:
            w.__enter__()
        try:
            for step in range(9):
                writer = writers[step % NUM_SERVERS]
                writer.append({"obs": np.array([step], dtype=np.float32)})
                writer.create_item(
                    table="experience",
                    priority=1.0,
                    trajectory={"obs": writer.history["obs"][-1:]},
                )

            for writer in writers:
                writer.flush()
        finally:
            for w in writers:
                w.__exit__(None, None, None)

        # Sample from each.
        for i, client in enumerate(clients):
            sample = next(
                client.sample("experience", num_samples=1, emit_timesteps=False)
            )
            val = np.asarray(sample.data[0]).reshape(-1)[0]
            print(f"  Server {i} sampled value: {val:.0f}")

        # Note: each server holds different data in this demo. In production
        # with a load balancer, the same experience may be written to all
        # servers for replication, or partitioned for capacity — Reverb does
        # not replicate automatically; the client decides where data goes.

    finally:
        for server in servers:
            server.stop()


# ---------------------------------------------------------------------------
# Example 2: Server.wait() — blocking until shutdown
# ---------------------------------------------------------------------------
def example_2_server_wait():
    """`Server.wait()` blocks the calling thread until the server is stopped
    (by another thread calling `stop()` or by SIGINT). This is the standard
    pattern for long-running server processes.

    Here we demonstrate it in a background thread.
    """
    print()
    print("=" * 60)
    print("Example 2: Server.wait()")
    print("=" * 60)

    server = reverb.Server(
        tables=[reverb.Table.queue(name="experience", max_size=100)],
        in_process=False,
    )

    def run_server():
        print("  Server thread started, waiting...")
        server.wait()  # blocks until stop()
        print("  Server thread unblocked (server stopped)")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    time.sleep(0.5)

    # Main thread stops the server, which unblocks wait().
    server.stop()
    thread.join(timeout=2)
    print("  Done.")


# ---------------------------------------------------------------------------
# Example 3: Server.localhost_client()
# ---------------------------------------------------------------------------
def example_3_localhost_client():
    """`Server.localhost_client()` is a convenience that creates a gRPC
    `Client` pointed at `localhost:<port>`. Only available in gRPC mode
    (`in_process=False`).
    """
    print()
    print("=" * 60)
    print("Example 3: localhost_client()")
    print("=" * 60)

    server = reverb.Server(
        tables=[reverb.Table.queue(name="experience", max_size=100)],
        in_process=False,
    )

    try:
        # Shorthand for: reverb.Client(f'localhost:{server.port}')
        client = server.localhost_client()
        client.insert(
            {"obs": np.array([42.0], dtype=np.float32)},
            priorities={"experience": 1.0},
        )
        sample = next(client.sample("experience", num_samples=1, emit_timesteps=False))
        print(f"  Sampled: {np.asarray(sample.data[0]).reshape(-1)[0]:.0f}")

    finally:
        server.stop()


# ===========================================================================
# Main
# ===========================================================================
def main():
    example_1_round_robin()
    example_2_server_wait()
    example_3_localhost_client()
    print()
    print("All examples passed.")


if __name__ == "__main__":
    main()
