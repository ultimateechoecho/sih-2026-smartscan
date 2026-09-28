"""Run every block's self-test in a fresh process:  python -m smartscan.selftest"""
import subprocess
import sys

MODULES = ["b1_data", "b1_emitters", "b2_env", "b3_agents", "b5_priority",
           "state_builder", "b6_scheduler", "metrics", "runner"]

if __name__ == "__main__":
    failed = []
    for m in MODULES:
        r = subprocess.run([sys.executable, "-m", f"smartscan.{m}"], capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  smartscan.{m}")
        if not ok:
            print(r.stdout[-600:], r.stderr[-1200:])
            failed.append(m)
    sys.exit(1 if failed else 0)
