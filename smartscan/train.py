"""TRAIN + VALIDATE - offline tuning of the Block 6 scheduler.

What is "trained": the scheduler's weights (exploration strength, forgetting rate, and the
weights on the due / priority / staleness features). Everything else (Blocks 3, 5) is
deterministic. In deployment there is no retraining: the bandit adapts within a mission
through its own hit/miss statistics.

Procedure
  1. sample `--trials` parameter sets (random search, first trial = defaults)
  2. score each on TRAIN scenarios  (episode seeds SEED_BASE['train']+i, train transmitter configs)
  3. re-score the best 3 on VALIDATION scenarios (unseen seeds + unseen transmitter configs)
  4. save the val-best parameters to models/best_<algo>.json and the log to results/train_log_<algo>.csv

    python -m smartscan.train --algo ucb --trials 24 --train-eps 5 --val-eps 8 --jobs 4
    python -m smartscan.train --algo ucb --quick        # 6 trials, 2/3 episodes: smoke test
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .b6_scheduler import SchedulerParams, make_scheduler
from .config import MODELS_DIR, RESULTS_DIR, SEED_BASE
from .metrics import aggregate, episode_metrics
from .runner import make_env, run_episode

_ENVS = {}


def _env(split):
    if split not in _ENVS:
        _ENVS[split] = make_env(split)
    return _ENVS[split]


def evaluate_params(args):
    """Mean metrics of `algo` with `params` on n episodes of `split` (top-level: picklable)."""
    algo, params_dict, split, n = args
    env = _env(split)
    params = SchedulerParams(**params_dict)
    rows = [episode_metrics(run_episode(env, make_scheduler(algo, params),
                                        SEED_BASE[split] + i)) for i in range(n)]
    agg = aggregate(rows)
    return {k: v[0] for k, v in agg.items()}


def sample_params(rng):
    return SchedulerParams(
        c=float(rng.uniform(0.0, 0.3)),
        gamma=float(rng.uniform(0.97, 0.997)),
        w_due=float(np.exp(rng.uniform(np.log(0.2), np.log(4.0)))),
        w_pri=float(rng.uniform(0.0, 1.5)),
        w_stale=float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))),
        w_hop=float(rng.uniform(0.0, 1.0)))


def _map(fn, jobs_list, jobs):
    if jobs <= 1:
        return [fn(j) for j in jobs_list]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, jobs_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ucb", choices=["ucb", "thompson"])
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--train-eps", type=int, default=5)
    ap.add_argument("--val-eps", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.trials, a.train_eps, a.val_eps = 6, 2, 3

    rng = np.random.default_rng(a.seed)
    cands = [SchedulerParams()] + [sample_params(rng) for _ in range(a.trials - 1)]
    t0 = time.time()
    print(f"[train] {a.algo}: {len(cands)} trials x {a.train_eps} train episodes (jobs={a.jobs})")
    tr = _map(evaluate_params, [(a.algo, c.__dict__, "train", a.train_eps) for c in cands], a.jobs)
    order = np.argsort([-r["avg_reward"] for r in tr])
    top = order[:3]
    print(f"[train] validating top {len(top)} on {a.val_eps} unseen validation episodes")
    va = _map(evaluate_params, [(a.algo, cands[i].__dict__, "val", a.val_eps) for i in top], a.jobs)
    best_k = int(np.argmax([r["avg_reward"] for r in va]))
    best_i = int(top[best_k])
    best = cands[best_i]

    fixed_val = evaluate_params(("fixed", {}, "val", a.val_eps))
    rows = []
    for i, (c, r) in enumerate(zip(cands, tr)):
        rows.append({**c.__dict__, **{f"train_{k}": r[k] for k in ("pd", "interception_ratio",
                                      "intercept_delay_cens_s", "avg_reward")}})
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"train_log_{a.algo}.csv", index=False)
    best.to_json(MODELS_DIR / f"best_{a.algo}.json")

    print(f"[train] done in {time.time()-t0:.0f}s. best params -> models/best_{a.algo}.json")
    print("        ", best)
    hdr = f"{'':14s}{'pd':>6s}{'ratio':>8s}{'delay_c(s)':>12s}{'reward':>9s}"
    print(hdr)
    for name, r in (("fixed (val)", fixed_val), (f"{a.algo} (val)", va[best_k])):
        print(f"{name:14s}{r['pd']:6.2f}{r['interception_ratio']:8.3f}"
              f"{r['intercept_delay_cens_s']:12.1f}{r['avg_reward']:9.1f}")


if __name__ == "__main__":
    main()
