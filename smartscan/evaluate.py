"""EVALUATE - compare schedulers on held-out scenarios with the PS figures of merit.

All algorithms run on the SAME episode seeds (paired comparison). Tuned parameters are
loaded from models/best_<algo>.json when present (run smartscan.train first), otherwise
defaults are used.

    python -m smartscan.evaluate --split test --episodes 30
    python -m smartscan.evaluate --split val  --episodes 10 --algos fixed random greedy ucb
Outputs: results/metrics_<split>.json, results/metrics_<split>.csv, results/compare_<split>.png
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .b6_scheduler import ALGOS, load_tuned, make_scheduler
from .config import RESULTS_DIR, SEED_BASE
from .metrics import ORDER, aggregate, episode_metrics
from .runner import make_env, run_episode

_ENV = {}


def _one(args):
    algo, split, seed = args
    if split not in _ENV:
        _ENV[split] = make_env(split)
    params = load_tuned(algo) if algo in ("ucb", "thompson") else None
    return algo, episode_metrics(run_episode(_ENV[split], make_scheduler(algo, params), seed))


def run_benchmark(algos, split="test", episodes=20, jobs=1):
    work = [(a, split, SEED_BASE[split] + i) for a in algos for i in range(episodes)]
    if jobs <= 1:
        res = [_one(w) for w in work]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            res = list(ex.map(_one, work))
    per_algo = {a: [m for x, m in res if x == a] for a in algos}
    return {a: aggregate(rows) for a, rows in per_algo.items()}, per_algo


def save_outputs(table, split, episodes):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"metrics_{split}.json").write_text(json.dumps(
        {"split": split, "episodes": episodes,
         "metrics": {a: {k: list(v) for k, v in m.items()} for a, m in table.items()}}, indent=2))
    df = pd.DataFrame({a: {k: m[k][0] for k in ORDER} for a, m in table.items()}).T
    df.to_csv(RESULTS_DIR / f"metrics_{split}.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [("pd", "Pd (emitters intercepted)"), ("interception_ratio", "Interception ratio (bursts)"),
                ("intercept_delay_cens_s", "Intercept delay (s, lower=better)"),
                ("avg_reward", "Avg episode reward")]
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
        names = list(table)
        for ax, (k, title) in zip(axs, keys):
            ax.bar(names, [table[a][k][0] for a in names], yerr=[table[a][k][1] for a in names],
                   color=["#999" if a in ("fixed", "random", "greedy") else "#1f77b4" for a in names],
                   capsize=3)
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"compare_{split}.png", dpi=130)
        plt.close(fig)
    except Exception as e:                                      # plotting is optional
        print("plot skipped:", e)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    table, _ = run_benchmark(a.algos, a.split, a.episodes, a.jobs)
    df = save_outputs(table, a.split, a.episodes)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print(f"\n=== {a.split} split, {a.episodes} paired episodes (mean) ===")
    print(df)
    print(f"\nsaved -> results/metrics_{a.split}.json|csv, compare_{a.split}.png")


if __name__ == "__main__":
    main()
