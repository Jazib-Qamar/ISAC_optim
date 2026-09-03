"""Run the ICC experiment suite, pytest, and paper-artifact generator.

Order: pytest → fairness/natural/mechanism → PSL/frontier/map/numerology/ablation → figures.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from experiments.icc_common import ICC_LOG, ICC_SUM, icc_dirs

EXPERIMENTS = [
    "experiments.icc_fair_comparison",
    "experiments.icc_natural_tdl_mc",
    "experiments.icc_controlled_asymmetry",
    "experiments.icc_cutting_plane_mc",
    "experiments.icc_frontier",
    "experiments.icc_feasibility_map",
    "experiments.icc_numerology",
    "experiments.icc_ablation",
    "experiments.icc_generate_paper",
]


def run(cmd: list[str], log_name: str | None = None) -> int:
    print(">>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/workspace")
    if log_name:
        (ICC_LOG / log_name).write_text(f"cmd={' '.join(cmd)}\nreturncode={proc.returncode}\n")
    return proc.returncode


def main() -> None:
    icc_dirs()
    extra = sys.argv[1:]  # e.g. --quick
    t0 = time.perf_counter()
    pytest_cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line"]
    print(">>>", " ".join(pytest_cmd), flush=True)
    pytest = subprocess.run(pytest_cmd, cwd="/workspace", capture_output=True, text=True)
    (ICC_SUM / "pytest.txt").write_text(pytest.stdout + "\n" + pytest.stderr)
    print(pytest.stdout)
    if pytest.returncode != 0:
        print(pytest.stderr)
        print("pytest failed; continuing to experiments so partial artifacts still exist", flush=True)

    rc = 0
    for mod in EXPERIMENTS:
        cmd = [sys.executable, "-m", mod, *extra]
        code = run(cmd, log_name=mod.replace(".", "_") + ".exit")
        rc = rc or code
        if code != 0:
            print(f"WARNING: {mod} exited {code}", flush=True)

    elapsed = time.perf_counter() - t0
    (ICC_SUM / "suite_runtime_s.txt").write_text(f"{elapsed:.3f}\n")
    print(f"ICC suite finished in {elapsed/60:.1f} min  rc={rc}", flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
