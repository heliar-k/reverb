"""Ticket ⑦: SHM vs gRPC vs in-process sample-path benchmark.

Measures the **sample path** (the design's core pain point: N repeated
decompressions per sample) for three transports on the same table/data:

  - gRPC loopback:  Server(tables, port=N)         + Client('localhost:N')
  - SHM:            Server(tables, in_process=True, shm=True) + ShmClient(path)
  - in-process:     Server(tables, in_process=True) + server.in_process_client
                    (upper bound: no serialization at all)

For each transport: warm up, then time M single-sample calls; report
throughput (samples/s) + p50/p99 per-sample latency.

// ponytail: plain `python reverb/benchmarks/shm_benchmark.py` after a
// successful `bazel build //reverb:pybind` — simpler than a bazel py_binary
// target and the pybind .so is already on the import path via the reverb
// package. Falls back to the same env setup the tests use.
"""

import time

import numpy as np

import reverb

TABLE = "bench"
WARMUP = 20  # samples discarded before timing


def _table(name=TABLE, max_size=20000):
    return reverb.Table(
        name=name,
        sampler=reverb.selectors.Fifo(),
        remover=reverb.selectors.Fifo(),
        max_size=max_size,
        max_times_sampled=0,  # reusable so we can draw more samples than inserted
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )


def _seed(client, n):
    """Seed `n` single-step items carrying one float32[1] column.

    Mirrors `_insert_one` in shm_test.py / in_process_test.py: a fresh
    trajectory_writer per item (num_keep_alive_refs=1). This is the proven
    pattern that works on all three transports (ShmClient.NewWriter is not
    implemented in v1, so `client.insert` can't be used for the SHM path).
    """
    val = np.array([1.0], dtype=np.float32)
    for _ in range(n):
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append({"v": val})
            w.create_item(
                table=TABLE, priority=1.0, trajectory={"v": w.history["v"][:]}
            )
            w.flush()


def _percentile(sorted_latencies, pct):
    if not sorted_latencies:
        return float("nan")
    k = (len(sorted_latencies) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_latencies) - 1)
    if f == c:
        return sorted_latencies[f]
    return sorted_latencies[f] + (sorted_latencies[c] - sorted_latencies[f]) * (k - f)


def _bench(client, num_samples):
    """Returns (throughput_sps, p50_s, p99_s). Times each sample() call."""
    # Warm up (discard).
    for _ in range(WARMUP):
        list(client.sample(TABLE, num_samples=1, emit_timesteps=False))

    latencies = []
    start = time.perf_counter()
    for _ in range(num_samples):
        t0 = time.perf_counter()
        list(client.sample(TABLE, num_samples=1, emit_timesteps=False))
        latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - start

    latencies.sort()
    return (
        num_samples / elapsed,
        _percentile(latencies, 50),
        _percentile(latencies, 99),
    )


def bench_shm(table_size, num_samples):
    server = reverb.Server(tables=[_table()], in_process=True, shm=True)
    try:
        path = server.shm_socket_path
        assert path, "shm_socket_path must be set when shm=True"
        client = reverb.ShmClient(path)
        _seed(client, table_size)
        return _bench(client, num_samples)
    finally:
        server.stop()


def bench_in_process(table_size, num_samples):
    server = reverb.Server(tables=[_table()], in_process=True)
    try:
        client = server.in_process_client
        _seed(client, table_size)
        return _bench(client, num_samples)
    finally:
        server.stop()


def _grpc_server_proc(port, table_size, ready, done):
    """Server process: owns the Table, serves real gRPC on `port`.

    Runs in a SEPARATE process so the client's `GetLocalTablePtr` PID check
    fails and the real gRPC SampleStream path is exercised (the same-process
    shortcut would otherwise bypass gRPC entirely, making this not a loopback
    measurement at all).
    """
    server = reverb.Server(
        tables=[_table(max_size=max(table_size * 2, 20000))],
        port=port,
        in_process=False,
    )
    ready.put(port)
    done.get()  # block until the main process signals shutdown
    server.stop()


def bench_grpc(table_size, num_samples):
    import multiprocessing as mp
    import portpicker

    port = portpicker.pick_unused_port()
    ready, done = mp.Queue(), mp.Queue()
    p = mp.Process(target=_grpc_server_proc, args=(port, table_size, ready, done))
    p.start()
    ready.get()  # wait for the server to be listening
    try:
        client = reverb.Client(f"localhost:{port}")
        _seed(client, table_size)
        return _bench(client, num_samples)
    finally:
        done.put(None)
        p.join(timeout=30)


TRANS = [
    ("in-process", bench_in_process),
    ("SHM", bench_shm),
    ("gRPC-loopback", bench_grpc),
]

# (table_size, num_samples) — vary both to show scaling.
SCENARIOS = [
    (1000, 200),
    (10000, 500),
]


def fmt(tput, p50, p99):
    return f"{tput:>10.0f} | {p50 * 1e6:>9.1f} | {p99 * 1e6:>9.1f}"


def main():
    results = {}  # (trans_name, table_size, num_samples) -> (tput, p50, p99)
    for table_size, num_samples in SCENARIOS:
        print(
            f"\n=== table_size={table_size} samples={num_samples} (warmup={WARMUP}) ==="
        )
        for name, fn in TRANS:
            tput, p50, p99 = fn(table_size, num_samples)
            results[(name, table_size, num_samples)] = (tput, p50, p99)
            print(
                f"  {name:<16} throughput={tput:>9.0f} sps  "
                f"p50={p50 * 1e6:>8.1f}us  p99={p99 * 1e6:>8.1f}us"
            )

    # Comparison table.
    print("\n" + "=" * 70)
    print("SUMMARY (throughput samples/s | p50 us | p99 us)")
    print("-" * 70)
    for table_size, num_samples in SCENARIOS:
        print(f"\n[table={table_size}, samples={num_samples}]")
        print(f"  {'transport':<16} {'sps':>10} | {'p50 us':>9} | {'p99 us':>9}")
        for name, _ in TRANS:
            tput, p50, p99 = results[(name, table_size, num_samples)]
            print(f"  {name:<16} {fmt(tput, p50, p99)}")

    # Emit a machine-readable line for the doc capture.
    print("\n# RAW (for docs/shm-benchmark.md):")
    for (name, ts, ns), (tput, p50, p99) in results.items():
        print(
            f"# {name}\ttable={ts}\tsamples={ns}\t"
            f"sps={tput:.1f}\tp50_us={p50 * 1e6:.2f}\tp99_us={p99 * 1e6:.2f}"
        )


if __name__ == "__main__":
    main()
