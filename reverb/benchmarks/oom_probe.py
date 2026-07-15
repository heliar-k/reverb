#!/usr/bin/env python3
"""Run one benchmark config in a subprocess and report RSS + exit code.

Used to confirm whether `bazel run //reverb/benchmarks:client_benchmark`
dies from OOM.  Pure stdlib — no psutil dependency.

One subprocess per config so memory never accumulates across combos.
The parent samples the child's VmRSS every 100ms and records the peak.
Exit code 137 (128 + SIGKILL) with RSS climbing near the end is the
classic OOM-killer signature.

Usage:
  python reverb/benchmarks/oom_probe.py -- <benchmark args, e.g. --quick>
  python reverb/benchmarks/oom_probe.py -- --mode pipeline --transport grpc \
        --payload large --table-size 10000 --duration 10
"""
import os
import resource
import signal
import subprocess
import sys
import time

INTERVAL = 0.1


def _rss_mb(pid):
    """Read VmRSS from /proc/<pid>/status. Returns MB or None."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # "VmRSS:     12345 kB"
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
        pass
    return None


def main():
    args = sys.argv[1:]
    if not args:
        args = ["--quick"]

    # Use the bazel-built stub (hermetic python with numpy/psutil/reverb),
    # not the system python which lacks those deps.
    target = os.environ.get("CLIENT_BENCH", "bazel-bin/reverb/benchmarks/client_benchmark")
    cmd = [target] + args
    print(f"# probe: {' '.join(args)}", flush=True)
    print(f"# probe: cmd={' '.join(cmd)}", flush=True)

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)

    peak_rss = 0.0
    samples = []
    killed_by_signal = None

    # Stream child output while sampling RSS.
    while True:
        line = proc.stdout.readline()
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
        rss = _rss_mb(proc.pid)
        if rss is not None:
            if rss > peak_rss:
                peak_rss = rss
            samples.append(rss)
        rc = proc.poll()
        if rc is not None:
            # drain remaining output
            rest = proc.stdout.read()
            if rest:
                sys.stdout.write(rest)
            break
        time.sleep(INTERVAL)

    elapsed = time.perf_counter() - t0

    if rc < 0:
        sig = -rc
        killed_by_signal = sig
        verdict = (f"KILLED by signal {sig} "
                   f"({signal.Signals(sig).name}) — likely OOM if SIGKILL(9)")
    elif rc == 137:
        verdict = "exit 137 (128+SIGKILL) — classic OOM-killer signature"
    else:
        verdict = f"normal exit rc={rc}"

    print(f"\n# ── probe summary ──", flush=True)
    print(f"# config:      {' '.join(args)}", flush=True)
    print(f"# exit code:   {rc}", flush=True)
    print(f"# verdict:     {verdict}", flush=True)
    print(f"# elapsed:     {elapsed:.1f}s", flush=True)
    print(f"# peak RSS:    {peak_rss:.1f} MB", flush=True)
    if samples:
        last = samples[-5:]
        print(f"# last 5 RSS:  {['%.0f' % s for s in last]} MB", flush=True)
        # climb in the last samples == grew until killed
        if len(samples) >= 5 and samples[-1] >= samples[len(samples)//2]:
            print(f"# note:        RSS still climbing near end", flush=True)

    # Also report this parent's own peak (ru_maxrss is in kB on Linux).
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f"# child ru_maxrss: {ru.ru_maxrss / 1024:.1f} MB", flush=True)

    # Exit nonzero if killed by signal, to make it scriptable.
    sys.exit(1 if (rc < 0 or rc == 137) else 0)


if __name__ == "__main__":
    main()
