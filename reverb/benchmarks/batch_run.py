#!/usr/bin/env python3
"""Run the client benchmark in batches, one subprocess per combo.

Why: `bazel run //reverb/benchmarks:client_benchmark` with the full matrix
gets SIGKILL'd by the cgroup OOM killer because memory accumulates across
combos inside one process (payload pools, reverb tables, grpc server
children are never released back to the OS within one run).

Fix: one subprocess per (transport, payload, table_size) combo.  Each
child dies after its combo, freeing all memory before the next starts.
JSON results from each child are collected and merged at the end.

Usage:
  # full matrix, batched
  python reverb/benchmarks/batch_run.py

  # subset (args pass through to client_benchmark)
  python reverb/benchmarks/batch_run.py --transport shm,grpc --payload large
  python reverb/benchmarks/batch_run.py --quick

Each child prints one JSON array on its last stdout line (via --json);
this script reassembles them.  If a child is killed (OOM/signal), its
exit code is recorded and the batch continues.
"""
import argparse
import json
import os
import signal
import subprocess
import sys

# Mirror client_benchmark's defaults so batch_run with no args == full matrix.
ALL_TRANSPORTS = ["inprocess", "shm", "grpc"]
ALL_PAYLOADS = ["small", "med", "large", "mixed"]
DEFAULT_TABLE_SIZES = [1000, 10000]
ALL_MODES = ["sample", "insert", "pipeline"]

TARGET = os.environ.get(
    "CLIENT_BENCH", "bazel-bin/reverb/benchmarks/client_benchmark"
)


def _parse_csv(val, all_vals):
    if val is None or val == "all":
        return all_vals
    return [x.strip() for x in val.split(",")]


def main():
    p = argparse.ArgumentParser(description="Batched client benchmark runner")
    p.add_argument("--mode", default="all")
    p.add_argument("--transport", default="all")
    p.add_argument("--payload", default="all")
    p.add_argument("--table-size", default=",".join(map(str, DEFAULT_TABLE_SIZES)))
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--num-ops", type=int, default=500)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--json", action="store_true",
                   help="emit merged JSON instead of the text report")
    args = p.parse_args()

    if args.quick:
        modes = ["sample", "pipeline"]
        transports = ALL_TRANSPORTS
        payloads = ["med"]
        table_sizes = [10000]
    else:
        modes = ALL_MODES if args.mode == "all" else [args.mode]
        transports = _parse_csv(args.transport, ALL_TRANSPORTS)
        payloads = _parse_csv(args.payload, ALL_PAYLOADS)
        table_sizes = [int(x) for x in args.table_size.split(",")]

    # Batch key = (payload, table_size): a child runs all transports+modes
    # for one payload pool, since the pool is the big memory item and it's
    # shared across transports inside one child.  One pool per child is fine;
    # what kills us is holding *several* pools at once.
    all_results = []
    failures = []

    for pl_spec in payloads:
        for table_size in table_sizes:
            child_args = [
                "--mode", "all" if len(modes) > 1 else modes[0],
                "--transport", ",".join(transports),
                "--payload", pl_spec,
                "--table-size", str(table_size),
                "--duration", str(args.duration),
                "--warmup", str(args.warmup),
                "--num-ops", str(args.num_ops),
                "--json",
            ]
            tag = f"{pl_spec}|{table_size}"
            print(f"\n# === batch {tag} ===", file=sys.stderr, flush=True)
            print(f"# cmd: {TARGET} {' '.join(child_args)}", file=sys.stderr, flush=True)

            proc = subprocess.run(
                [TARGET] + child_args,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

            if proc.returncode != 0:
                sig = -proc.returncode if proc.returncode < 0 else None
                reason = (f"signal {sig} ({signal.Signals(sig).name})"
                          if sig else f"exit {proc.returncode}")
                # OOM leaves a recognizable pattern: nonzero, empty/short stdout.
                print(f"# batch {tag} FAILED ({reason})", file=sys.stderr, flush=True)
                print(f"# stderr tail:\n{proc.stderr[-800:]}", file=sys.stderr, flush=True)
                failures.append((tag, reason))
                continue

            try:
                child_results = json.loads(proc.stdout)
                all_results.extend(child_results)
                print(f"# batch {tag} ok: {len(child_results)} results",
                      file=sys.stderr, flush=True)
            except json.JSONDecodeError as e:
                print(f"# batch {tag}: could not parse JSON ({e})",
                      file=sys.stderr, flush=True)
                print(f"# stdout tail:\n{proc.stdout[-800:]}",
                      file=sys.stderr, flush=True)
                failures.append((tag, f"json parse: {e}"))

    print(f"\n# === summary ===", file=sys.stderr, flush=True)
    print(f"# collected {len(all_results)} results from "
          f"{len(payloads) * len(table_sizes)} batches, "
          f"{len(failures)} failures", file=sys.stderr, flush=True)
    for tag, reason in failures:
        print(f"#   FAIL {tag}: {reason}", file=sys.stderr, flush=True)

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        # Reuse client_benchmark's text report by feeding merged JSON back in
        # is overkill; just dump compact lines.
        for r in all_results:
            parts = [f"{k}={v}" for k, v in r.items()]
            print("# " + "\t".join(parts))

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
