#!/usr/bin/env python3
"""Measure total cgroup memory while running a benchmark config.

Reads /sys/fs/cgroup/memory/memory.usage_in_bytes (cgroup v1) — this
captures the WHOLE process tree including the grpc server grandchild,
which ru_maxrss on the parent misses.

Prints peak cgroup usage in GB alongside exit code.
"""
import subprocess
import sys
import threading
import time

CGROUP_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
TARGET = "bazel-bin/reverb/benchmarks/client_benchmark"


def main():
    args = sys.argv[1:] or ["--quick"]
    print(f"# cg-probe: {TARGET} {' '.join(args)}", flush=True)

    proc = subprocess.Popen([TARGET] + args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    peak = 0
    samples = []
    while proc.poll() is None:
        try:
            with open(CGROUP_USAGE) as f:
                b = int(f.read().strip())
            gb = b / 1024**3
            if gb > peak:
                peak = gb
            samples.append(gb)
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.2)
    rc = proc.wait()

    print(f"# exit: {rc}", flush=True)
    print(f"# peak cgroup mem: {peak:.2f} GB", flush=True)
    if samples:
        print(f"# samples: {len(samples)}, "
              f"start={samples[0]:.2f}GB end={samples[-1]:.2f}GB", flush=True)
    sys.exit(1 if (rc < 0 or rc == 137) else 0)


if __name__ == "__main__":
    main()
