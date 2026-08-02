#!/usr/bin/env python3
"""Unified Reverb client benchmark: sample, insert, and pipeline paths.

Covers all three transports (in-process, SHM, gRPC loopback) across
multiple payload sizes, table sizes, and concurrency models.

Usage:
  python reverb/benchmarks/client_benchmark.py              # full matrix
  python reverb/benchmarks/client_benchmark.py --quick      # fast comparison
  python reverb/benchmarks/client_benchmark.py --transport shm,grpc --mode pipeline
  python reverb/benchmarks/client_benchmark.py --mode pipeline --duration 60  # stability
  python reverb/benchmarks/client_benchmark.py --json       # machine-readable
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np

import reverb

# ── resource monitoring ──────────────────────────────────────────────────────


class _ResourceMonitor(threading.Thread):
    """Daemon thread sampling RSS/CPU of a process every 100ms."""

    def __init__(self, pid, interval=0.1):
        super().__init__(daemon=True)
        self._pid = pid
        self._interval = interval
        self._done = threading.Event()
        self._rss_mb = []
        self._cpu_pct = []

    def run(self):
        import psutil

        try:
            proc = psutil.Process(self._pid)
            proc.cpu_percent()
            time.sleep(self._interval)
            while not self._done.is_set():
                self._rss_mb.append(proc.memory_info().rss / (1024 * 1024))
                self._cpu_pct.append(proc.cpu_percent())
                time.sleep(self._interval)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def stop_and_report(self):
        self._done.set()
        self.join(timeout=2)
        if not self._rss_mb:
            return {}
        return {
            "peak_rss_mb": max(self._rss_mb),
            "avg_cpu_pct": sum(self._cpu_pct) / max(len(self._cpu_pct), 1),
        }


def _has_psutil():
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


# ── helpers ───────────────────────────────────────────────────────────────────

TABLE = "bench"


def _make_table(name=TABLE, max_size=20000):
    return reverb.Table(
        name=name,
        sampler=reverb.selectors.Fifo(),
        remover=reverb.selectors.Fifo(),
        max_size=max_size,
        max_times_sampled=0,
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )


def _percentile(sorted_data, pct):
    # 输入已排序;np.percentile 的 linear 插值与原手搓实现语义一致。
    return float(np.percentile(sorted_data, pct)) if sorted_data else float("nan")


# ── payload generation ───────────────────────────────────────────────────────

# ponytail: pre-generate N copies of each shape in a buffer pool,
#          benchmark just indexes into the pool — no randn in hot path.

_SHAPES = {
    "small": (1,),  # float32[1]           ≈    4 B
    "med": (84, 84, 4),  # float32[84,84,4]     ≈  110 KB
    "large": (256, 256, 3),  # float32[256,256,3]   ≈  770 KB
}


def _payload_pool(spec, n):
    """Return a list of `n` payload dicts {"v": ndarray}."""
    if spec == "mixed":
        bufs = {k: np.random.randn(*s).astype(np.float32) for k, s in _SHAPES.items()}
        keys = list(bufs.keys())
        return [{"v": bufs[keys[i % 3]].copy()} for i in range(n)]
    shape = _SHAPES[spec]
    buf = np.random.randn(*shape).astype(np.float32)
    return [{"v": buf.copy()} for _ in range(n)]


# ── seed table ────────────────────────────────────────────────────────────────


def _seed(client, n, payloads):
    """Insert `n` single-step items into TABLE via trajectory_writer."""
    for i in range(n):
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            w.append(payloads[i % len(payloads)])
            w.create_item(
                table=TABLE,
                priority=1.0,
                trajectory={"v": w.history["v"][:]},
            )
            w.flush()


# ── server / client setup ────────────────────────────────────────────────────


def _setup_inprocess(table_size, payloads):
    server = reverb.Server(
        tables=[_make_table(max_size=max(table_size * 2, 20000))],
        in_process=True,
    )
    client = server.in_process_client
    _seed(client, table_size, payloads)
    return server, client, os.getpid()


def _setup_shm(table_size, payloads):
    server = reverb.Server(
        tables=[_make_table(max_size=max(table_size * 2, 20000))],
        in_process=True,
        shm=True,
    )
    path = server.shm_socket_path
    assert path, "shm_socket_path must be set when shm=True"
    client = reverb.ShmClient(path)
    _seed(client, table_size, payloads)
    return server, client, os.getpid()


def _grpc_server_proc(port, table_size, ready, done):
    server = reverb.Server(
        tables=[_make_table(max_size=max(table_size * 2, 20000))],
        port=port,
        in_process=False,
    )
    ready.put((port, os.getpid()))
    done.get()
    server.stop()


def _setup_grpc(table_size, payloads):
    import portpicker

    port = portpicker.pick_unused_port()
    ready, done = mp.Queue(), mp.Queue()
    p = mp.Process(
        target=_grpc_server_proc,
        args=(port, table_size, ready, done),
    )
    p.start()
    try:
        port_ok, server_pid = ready.get(timeout=30)
        assert port_ok == port, f"expected port {port}, got {port_ok}"
    except Exception:
        p.terminate()
        p.join(timeout=5)
        raise
    client = reverb.Client(f"localhost:{port}")
    _seed(client, table_size, payloads)
    return (p, done), client, server_pid, port


SETUP_FNS = {
    "inprocess": _setup_inprocess,
    "shm": _setup_shm,
    "grpc": _setup_grpc,
}

TEARDOWN_FNS = {
    "inprocess": lambda s, _c, _p, _h: s.stop(),
    "shm": lambda s, _c, _p, _h: s.stop(),
    "grpc": lambda _s, _c, _p, h: _teardown_grpc(h),
}


def _teardown_grpc(handles):
    proc, done_q = handles
    done_q.put(None)
    proc.join(timeout=30)


# ── benchmark: sample ────────────────────────────────────────────────────────


def _bench_sample(client, num_samples, warmup):
    """Time `num_samples` single-sample calls. Returns (tput, sorted_latencies)."""
    for _ in range(warmup):
        list(client.sample(TABLE, num_samples=1, emit_timesteps=False))

    latencies = []
    start = time.perf_counter()
    for _ in range(num_samples):
        t0 = time.perf_counter()
        list(client.sample(TABLE, num_samples=1, emit_timesteps=False))
        latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - start

    return num_samples / elapsed if elapsed > 0 else 0, sorted(latencies)


# ── benchmark: insert ────────────────────────────────────────────────────────


def _bench_insert(client, num_inserts, payloads, warmup, batch=1):
    """Time `num_inserts` item inserts, `batch` items per flush.

    Returns (items/s, sorted per-flush latencies). With batch>1 the latency
    is per flush (batch items), not per item.
    """

    def write_batch(start):
        with client.trajectory_writer(num_keep_alive_refs=1) as w:
            for j in range(batch):
                w.append(payloads[(start + j) % len(payloads)])
                # [-1:] not [:]: keep_alive=1 expires older refs to None as the
                # batch grows, and TrajectoryColumn rejects None refs.
                w.create_item(
                    table=TABLE, priority=1.0, trajectory={"v": w.history["v"][-1:]}
                )
            w.flush()

    for i in range(0, warmup * batch, batch):
        write_batch(i)

    latencies = []
    start = time.perf_counter()
    for i in range(0, num_inserts, batch):
        t0 = time.perf_counter()
        write_batch(i)
        latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - start

    return num_inserts / elapsed if elapsed > 0 else 0, sorted(latencies)


# ── benchmark: pipeline ──────────────────────────────────────────────────────


def _bench_pipeline(
    client, duration_s, payloads, server_pid, writers=1, readers=1, client_factory=None
):
    """Concurrent insert + sample for `duration_s`. Returns dict of metrics.

    `client` serves the first writer/reader thread. Extra writers/readers get
    their own connection from `client_factory` (None -> all threads share
    `client`). Thread-safe without extra locking: list.append is atomic under
    the GIL, and each transport client releases the GIL in pybind calls.
    """
    payloads = list(payloads)
    n_payloads = len(payloads)

    def extra_clients(n):
        if n <= 1 or client_factory is None:
            return [client] * n
        return [client] + [client_factory() for _ in range(n - 1)]

    w_clients = extra_clients(writers)
    r_clients = extra_clients(readers)

    insert_latencies = []
    sample_latencies = []
    insert_count = [0]
    sample_count = [0]
    stop = threading.Event()

    _seed(client, 20, payloads)

    def writer(cli):
        i = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            with cli.trajectory_writer(num_keep_alive_refs=1) as w:
                w.append(payloads[i % n_payloads])
                w.create_item(
                    table=TABLE, priority=1.0, trajectory={"v": w.history["v"][:]}
                )
                w.flush()
            insert_latencies.append(time.perf_counter() - t0)
            insert_count[0] += 1
            i += 1

    def reader(cli):
        while not stop.is_set():
            t0 = time.perf_counter()
            list(cli.sample(TABLE, num_samples=1, emit_timesteps=False))
            sample_latencies.append(time.perf_counter() - t0)
            sample_count[0] += 1

    monitor = _ResourceMonitor(server_pid) if _has_psutil() else None
    if monitor:
        monitor.start()

    threads = [
        threading.Thread(target=writer, args=(c,), daemon=True) for c in w_clients
    ] + [threading.Thread(target=reader, args=(c,), daemon=True) for c in r_clients]
    for t in threads:
        t.start()

    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    resource = monitor.stop_and_report() if monitor else {}

    insert_latencies.sort()
    sample_latencies.sort()

    elapsed = duration_s
    return {
        "insert_tput": insert_count[0] / elapsed if elapsed > 0 else 0,
        "sample_tput": sample_count[0] / elapsed if elapsed > 0 else 0,
        "insert_p50": _percentile(insert_latencies, 50),
        "insert_p90": _percentile(insert_latencies, 90),
        "insert_p99": _percentile(insert_latencies, 99),
        "sample_p50": _percentile(sample_latencies, 50),
        "sample_p90": _percentile(sample_latencies, 90),
        "sample_p99": _percentile(sample_latencies, 99),
        **resource,
    }


# ── runner ────────────────────────────────────────────────────────────────────


def _die(*msgs):
    print("\n".join(msgs), file=sys.stderr)
    sys.exit(1)


def _fmt_lat(sec):
    if sec != sec:  # NaN
        return "       nan"
    return f"{sec * 1e6:>8.0f}us"


def main():
    p = argparse.ArgumentParser(description="Unified Reverb client benchmark")
    p.add_argument(
        "--mode",
        default="all",
        choices=["sample", "insert", "pipeline", "all"],
    )
    p.add_argument(
        "--transport",
        default="all",
        help="comma-separated: inprocess,shm,grpc,all",
    )
    p.add_argument(
        "--payload",
        default="all",
        help="comma-separated: small,med,large,mixed,all",
    )
    p.add_argument(
        "--table-size",
        default="1000,10000",
        help="comma-separated integers",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=5,
        help="pipeline duration in seconds (use 60+ for stability check)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="fast comparison: sample+pipeline, med payload, 10k table, 5s",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="output machine-readable JSON",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="warmup ops for sample/insert modes",
    )
    p.add_argument(
        "--writers",
        default="1",
        help="comma-separated writer-thread counts for pipeline (zipped with --readers)",
    )
    p.add_argument(
        "--readers",
        default="1",
        help="comma-separated reader-thread counts for pipeline (zipped with --writers)",
    )
    p.add_argument(
        "--batch",
        default="1",
        help="comma-separated items-per-flush counts for insert mode",
    )
    p.add_argument(
        "--num-ops",
        type=int,
        default=500,
        help="number of operations for sample/insert modes",
    )
    args = p.parse_args()

    if args.quick:
        modes = ["sample", "pipeline"]
        transports = ["inprocess", "shm", "grpc"]
        payloads = ["med"]
        table_sizes = [10000]
    else:
        modes = ["sample", "insert", "pipeline"] if args.mode == "all" else [args.mode]
        transports = (
            ["inprocess", "shm", "grpc"]
            if args.transport == "all"
            else [t.strip() for t in args.transport.split(",")]
        )
        payloads = (
            ["small", "med", "large", "mixed"]
            if args.payload == "all"
            else [s.strip() for s in args.payload.split(",")]
        )
        table_sizes = [int(x.strip()) for x in args.table_size.split(",")]

    writer_counts = [int(x.strip()) for x in args.writers.split(",")]
    reader_counts = [int(x.strip()) for x in args.readers.split(",")]
    if len(writer_counts) != len(reader_counts):
        _die(
            "--writers and --readers must have the same number of entries (zipped pairs)"
        )
    batches = [int(x.strip()) for x in args.batch.split(",")]

    for t in transports:
        if t not in SETUP_FNS:
            _die(f"unknown transport: {t}")
    for pl in payloads:
        if pl not in ("small", "med", "large", "mixed"):
            _die(f"unknown payload: {pl}")
    if not _has_psutil():
        print("# psutil not installed — resource monitoring skipped", file=sys.stderr)

    results = []

    for table_size in table_sizes:
        for pl_spec in payloads:
            # ponytail: cap the pool for big payloads — items may repeat buffers,
            # table content is what matters; this box is a 16GiB k8s pod
            # (cgroup v1), 10k x 770KB (~8GB) would OOM-kill the run.
            pool_n = max(table_size, args.num_ops) + args.warmup
            pool_cap = {"med": 2000, "large": 500, "mixed": 1500}.get(pl_spec)
            if pool_cap is not None:
                pool_n = min(pool_n, pool_cap)
            pl_buf = _payload_pool(pl_spec, pool_n)

            for transport in transports:
                server, client, server_pid, cleanup_handle, factory = _do_setup(
                    transport, table_size, pl_buf
                )

                try:
                    for mode in modes:
                        base = f"{transport}|{pl_spec}|{table_size}"
                        key = f"{base}|{mode}"

                        if mode == "sample":
                            tput, lats = _bench_sample(
                                client, args.num_ops, args.warmup
                            )
                            results.append(
                                {
                                    "key": key,
                                    "tput_sps": tput,
                                    "p50_s": _percentile(lats, 50),
                                    "p90_s": _percentile(lats, 90),
                                    "p99_s": _percentile(lats, 99),
                                }
                            )

                        elif mode == "insert":
                            for batch in batches:
                                tput, lats = _bench_insert(
                                    client, args.num_ops, pl_buf, args.warmup, batch
                                )
                                results.append(
                                    {
                                        "key": f"{base}|b{batch}|insert",
                                        "tput_ips": tput,
                                        "p50_s": _percentile(lats, 50),
                                        "p90_s": _percentile(lats, 90),
                                        "p99_s": _percentile(lats, 99),
                                    }
                                )

                        elif mode == "pipeline":
                            for nw, nr in zip(writer_counts, reader_counts):
                                res = _bench_pipeline(
                                    client,
                                    args.duration,
                                    pl_buf,
                                    server_pid,
                                    writers=nw,
                                    readers=nr,
                                    client_factory=factory,
                                )
                                results.append(
                                    {"key": f"{base}|w{nw}r{nr}|pipeline", **res}
                                )

                except Exception as e:  # keep the matrix going; report the hole
                    results.append(
                        {
                            "key": f"{transport}|{pl_spec}|{table_size}|ERROR",
                            "error": repr(e),
                        }
                    )

                finally:
                    TEARDOWN_FNS[transport](server, client, server_pid, cleanup_handle)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_report(results)
        # Machine-readable raw lines
        print("\n# RAW")
        for r in results:
            parts = [f"{k}={v}" for k, v in r.items()]
            print("# " + "\t".join(parts))


def _do_setup(transport, table_size, pl_buf):
    """Returns (server, client, server_pid, cleanup_handle, client_factory).

    client_factory() opens a NEW connection to the same server (None for
    in-process, where threads share the single LocalClient safely).
    """
    if transport == "grpc":
        (proc, done_q), client, server_pid, port = _setup_grpc(table_size, pl_buf)
        factory = lambda: reverb.Client(f"localhost:{port}")  # noqa: E731
        return None, client, server_pid, (proc, done_q), factory
    server, client, server_pid = SETUP_FNS[transport](table_size, pl_buf)
    if transport == "shm":
        path = server.shm_socket_path
        factory = lambda: reverb.ShmClient(path)  # noqa: E731
    else:
        factory = None
    return server, client, server_pid, None, factory


# ── output formatting ─────────────────────────────────────────────────────────


def _print_report(results):
    errors = [r for r in results if "error" in r]
    results = [r for r in results if "error" not in r]
    for mode_label in ["sample", "insert", "pipeline"]:
        group = [r for r in results if r["key"].endswith("|" + mode_label)]
        if not group:
            continue

        print(f"\n{'=' * 90}")
        print(f"  MODE: {mode_label}")
        print(f"{'=' * 90}")

        if mode_label == "sample":
            print(
                f"  {'transport/payload/table':<40} "
                f"{'sps':>8}  {'p50':>10}  {'p90':>10}  {'p99':>10}"
            )
            print(f"  {'-' * 40} {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
            for r in sorted(group, key=lambda x: x["key"]):
                tag = r["key"].replace("|sample", "")
                print(
                    f"  {tag:<40} {r['tput_sps']:>8.0f}  "
                    f"{_fmt_lat(r['p50_s'])}  {_fmt_lat(r['p90_s'])}  {_fmt_lat(r['p99_s'])}"
                )

        elif mode_label == "insert":
            print(
                f"  {'transport/payload/table':<40} "
                f"{'ips':>8}  {'p50':>10}  {'p90':>10}  {'p99':>10}"
            )
            print(f"  {'-' * 40} {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
            for r in sorted(group, key=lambda x: x["key"]):
                tag = r["key"].replace("|insert", "")
                print(
                    f"  {tag:<40} {r['tput_ips']:>8.0f}  "
                    f"{_fmt_lat(r['p50_s'])}  {_fmt_lat(r['p90_s'])}  {_fmt_lat(r['p99_s'])}"
                )

        elif mode_label == "pipeline":
            print(
                f"  {'transport/payload/table':<40} "
                f"{'ins/s':>7} {'sam/s':>7} | "
                f"{'ins p50':>8} {'ins p90':>8} {'ins p99':>8} | "
                f"{'sam p50':>8} {'sam p90':>8} {'sam p99':>8}"
            )
            # only show rss/cpu if present
            has_res = any("peak_rss_mb" in r for r in group)
            if has_res:
                print(  # continuation of header
                    f"  {'':40} {'':7} {'':7} | {'':8} {'':8} {'':8} | {'':8} {'':8} {'':8} | "
                    f"{'rss_mb':>7} {'cpu%':>5}"
                )
            print(
                f"  {'-' * 40} {'-' * 7} {'-' * 7} | {'-' * 8} {'-' * 8} {'-' * 8} | {'-' * 8} {'-' * 8} {'-' * 8}"
            )
            for r in sorted(group, key=lambda x: x["key"]):
                tag = r["key"].replace("|pipeline", "")
                line = (
                    f"  {tag:<40} "
                    f"{r['insert_tput']:>7.0f} {r['sample_tput']:>7.0f} | "
                    f"{_fmt_lat(r['insert_p50'])} {_fmt_lat(r['insert_p90'])} {_fmt_lat(r['insert_p99'])} | "
                    f"{_fmt_lat(r['sample_p50'])} {_fmt_lat(r['sample_p90'])} {_fmt_lat(r['sample_p99'])}"
                )
                if has_res:
                    rss = r.get("peak_rss_mb", float("nan"))
                    cpu = r.get("avg_cpu_pct", float("nan"))
                    line += f" | {rss:>7.1f} {cpu:>5.1f}"
                print(line)

    if errors:
        print(f"\n{'=' * 90}\n  ERRORS ({len(errors)} configs skipped)")
        for r in errors:
            print(f"  {r['key']}: {r['error']}")


if __name__ == "__main__":
    main()
