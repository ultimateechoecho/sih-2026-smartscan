"""BLOCK 6 - The scheduler (the only tuned component).

Input : features (n_bands, 8) from StateBuilder, glob {"hop_conf"}, current slot t
Output: action = band index to dwell on next

Schedulers share one interface:  reset(n_bands, dwell_slots, seed) / select(F, glob, t) / update(a, hit, r_obs)
    FixedSweep, RandomScan, GreedyScan   -> baselines
    UCBScheduler, ThompsonScheduler      -> proposed (discounted bandits, non-stationary safe)

Score of band b (proposed):
    Q_b  +  explore_b  +  w_due * due_b  +  w_pri * priority_b  +  w_stale * visit_age_b
  Q_b        discounted mean observable reward per slot (or Thompson sample of hit prob)
  explore_b  c * (1 + w_hop * hop_conf) * sqrt(ln N / n_b)         (UCB only)
  due_b      periodicity/stagger/beam-scan agents say "this band is about to be active"
  priority_b Block 5 surprise/starvation score
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import MODELS_DIR


@dataclass
class SchedulerParams:
    c: float = 0.05          # UCB exploration strength
    gamma: float = 0.985     # per-step discount (forgetting) of bandit statistics
    w_due: float = 1.0       # revisit bands the agents say are about to be active (tracking)
    w_pri: float = 0.3       # bands that just behaved unexpectedly (Block 5)
    w_stale: float = 1.0     # coverage: bands not visited for a while (sweep-like search)
    w_hop: float = 0.5       # extra exploration when frequency hopping is suspected

    def to_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path):
        return SchedulerParams(**json.loads(Path(path).read_text()))


# ------------------------------------------------------------------ baselines
class BaseScheduler:
    name = "base"

    def reset(self, n_bands, dwell_slots, seed=0):
        self.n, self.dwell = n_bands, np.asarray(dwell_slots)
        self.rng = np.random.default_rng(seed)

    def select(self, F, glob, t):
        raise NotImplementedError

    def update(self, a, hit, r_obs):
        pass


class FixedSweep(BaseScheduler):
    """Open-loop: 0,1,2,...,35,0,1,... (the strategy the PS calls inefficient)."""
    name = "fixed"

    def reset(self, n_bands, dwell_slots, seed=0):
        super().reset(n_bands, dwell_slots, seed)
        self.i = -1

    def select(self, F, glob, t):
        self.i = (self.i + 1) % self.n
        return self.i


class RandomScan(BaseScheduler):
    name = "random"

    def select(self, F, glob, t):
        return int(self.rng.integers(self.n))


class GreedyScan(BaseScheduler):
    """Revisit the band with the best recent hit rate; 10% random; unseen bands first."""
    name = "greedy"

    def __init__(self, eps=0.1):
        self.eps = eps

    def select(self, F, glob, t):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n))
        unseen = np.flatnonzero(F[:, 1] >= 1.0)            # never/long unvisited
        if len(unseen) and F[:, 0].max() < 0.2:
            return int(self.rng.choice(unseen))
        return int(np.argmax(F[:, 0] + 1e-6 * self.rng.random(self.n)))


# ------------------------------------------------------------- proposed (bandit)
class _Bandit(BaseScheduler):
    def __init__(self, params: SchedulerParams = None):
        self.p = params or SchedulerParams()

    def reset(self, n_bands, dwell_slots, seed=0):
        super().reset(n_bands, dwell_slots, seed)
        self.cnt = np.zeros(n_bands)         # discounted visit count
        self.sum = np.zeros(n_bands)         # discounted reward (UCB) / hits (Thompson)

    def _bonus(self, F, glob):
        p = self.p
        return p.w_due * F[:, 4] + p.w_pri * F[:, 7] + p.w_stale * F[:, 1]

    def update(self, a, hit, r_obs):
        self.cnt *= self.p.gamma
        self.sum *= self.p.gamma
        self.cnt[a] += 1.0
        self.sum[a] += self._target(hit, r_obs)


class UCBScheduler(_Bandit):
    name = "ucb"

    def _target(self, hit, r_obs):
        return r_obs

    def select(self, F, glob, t):
        p = self.p
        n = self.cnt + 1e-2
        Q = self.sum / n
        N = max(2.0, self.cnt.sum() + 1.0)
        explore = p.c * (1.0 + p.w_hop * glob["hop_conf"]) * np.sqrt(np.log(N) / n)
        score = Q + np.minimum(explore, 6 * p.c) + self._bonus(F, glob)
        return int(np.argmax(score + 1e-9 * self.rng.random(self.n)))


class ThompsonScheduler(_Bandit):
    name = "thompson"

    def _target(self, hit, r_obs):
        return float(hit)

    def select(self, F, glob, t):
        alpha = 1.0 + self.sum
        beta = 1.0 + np.maximum(self.cnt - self.sum, 0.0)
        sample = self.rng.beta(alpha, beta)
        score = sample * self.p.c * 4 + self._bonus(F, glob)
        return int(np.argmax(score))


ALGOS = ("fixed", "random", "greedy", "ucb", "thompson")


def make_scheduler(name, params: SchedulerParams = None):
    if name == "fixed":
        return FixedSweep()
    if name == "random":
        return RandomScan()
    if name == "greedy":
        return GreedyScan()
    if name == "ucb":
        return UCBScheduler(params)
    if name == "thompson":
        return ThompsonScheduler(params)
    raise ValueError(f"unknown algo {name}; choose from {ALGOS}")


def load_tuned(name):
    """Tuned params from models/best_<algo>.json if train.py has been run, else defaults."""
    p = MODELS_DIR / f"best_{name}.json"
    return SchedulerParams.from_json(p) if p.exists() else SchedulerParams()


if __name__ == "__main__":
    n = 8
    F = np.zeros((n, 8))
    F[:, 1] = 0.5
    glob = {"hop_conf": 0.0}
    for name in ALGOS:
        s = make_scheduler(name)
        s.reset(n, np.ones(n, int), seed=0)
        acts = []
        for t in range(40):
            a = s.select(F, glob, t)
            assert 0 <= a < n
            s.update(a, hit=int(a == 3), r_obs=float(a == 3))
            acts.append(a)
        print(f"{name:9s} picks band 3 in {acts.count(3):2d}/40 steps")
    s = make_scheduler("ucb")
    s.reset(n, np.ones(n, int), seed=0)
    for t in range(200):
        a = s.select(F, glob, t)
        s.update(a, hit=int(a == 3), r_obs=float(a == 3))
    assert s.cnt.argmax() == 3, "UCB should learn to favour the rewarding band"
    F[5, 4] = 1.0                                          # "due" hint must attract a scan
    fresh = make_scheduler("ucb")
    fresh.reset(n, np.ones(n, int), seed=0)
    assert fresh.select(F, glob, 0) == 5, "a band that is 'due' should win on a fresh scheduler"
    print("b6_scheduler OK")
