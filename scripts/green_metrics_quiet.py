#!/usr/bin/env python3
"""Wait for ambient CPU load to fall below threshold, then run green_metrics.

The power model attributes ALL system CPU load to the measured fit, so the
green-metrics run must execute on a quiet machine. This watcher polls the
system-wide load and launches the measurement only after 3 consecutive quiet
checks (ambient average < 12%). Gives up after 2 hours.
"""
import subprocess
import sys
import time

import psutil

MIN_QUIET_CHECKS = 3
TIMEOUT_S = 2 * 60 * 60


def ambient_avg(window_s=3.0):
    n = max(1, int(window_s / 0.5))
    psutil.cpu_percent(interval=None)
    vals = [psutil.cpu_percent(interval=0.5) for _ in range(n)]
    return sum(vals) / len(vals)


def main():
    checks = 0
    t0 = time.time()
    while checks < MIN_QUIET_CHECKS:
        if time.time() - t0 > TIMEOUT_S:
            print("WATCHER TIMEOUT — machine never went quiet", flush=True)
            sys.exit(1)
        avg = ambient_avg()
        print(f"[{time.strftime('%H:%M:%S')}] ambient={avg:.1f}% "
              f"({checks}/{MIN_QUIET_CHECKS} quiet checks)", flush=True)
        checks = checks + 1 if avg < 12.0 else 0
        if checks < MIN_QUIET_CHECKS:
            time.sleep(20)
    print("Machine quiet — launching green metrics", flush=True)
    r = subprocess.run([sys.executable, "-u", "src/green_metrics.py"])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()