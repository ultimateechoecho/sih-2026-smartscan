"""Decision tracing for the demo frontend.

Wraps the Block 6 schedulers so every selection is recorded with its full score
breakdown (per band).  The math is copied verbatim from b6_scheduler.select — if
that file changes, keep these in sync.  Baselines get a human-readable reason
instead of score components (they have none: they are open-loop).

run_episode_traced() reuses runner.run_episode (and its new on_step hook) so the
episode loop exists exactly once.
"""
import numpy as np

from smartscan.b6_scheduler import (FixedSweep, GreedyScan, RandomScan,
                                    ThompsonScheduler, UCBScheduler)
from smartscan.runner import run_episode


class TracedUCB(UCBScheduler):
    """UCBScheduler that records Q / explore / due / priority / stale per band."""

    name = "ucb"
    traced = True

    def __init__(self, params=None):
        super().__init__(params)
        self.trace = []

    def select(self, F, glob, t):
        p = self.p
        n = self.cnt + 1e-2
        Q = self.sum / n
        N = max(2.0, self.cnt.sum() + 1.0)
        explore = np.minimum(
            p.c * (1.0 + p.w_hop * glob["hop_conf"]) * np.sqrt(np.log(N) / n), 6 * p.c)
        due = p.w_due * F[:, 4]
        pri = p.w_pri * F[:, 7]
        stale = p.w_stale * F[:, 1]
        score = Q + explore + due + pri + stale
        a = int(np.argmax(score + 1e-9 * self.rng.random(self.n)))
        self.trace.append({"t": t, "action": a, "hop_conf": glob["hop_conf"],
                           "Q": Q.tolist(), "explore": explore.tolist(),
                           "due": due.tolist(), "priority": pri.tolist(),
                           "stale": stale.tolist(), "score": score.tolist()})
        return a


class TracedThompson(ThompsonScheduler):
    """ThompsonScheduler that records the posterior sample + bonus per band."""

    name = "thompson"
    traced = True

    def __init__(self, params=None):
        super().__init__(params)
        self.trace = []

    def select(self, F, glob, t):
        p = self.p
        alpha = 1.0 + self.sum
        beta = 1.0 + np.maximum(self.cnt - self.sum, 0.0)
        sample = self.rng.beta(alpha, beta)
        due = p.w_due * F[:, 4]
        pri = p.w_pri * F[:, 7]
        stale = p.w_stale * F[:, 1]
        score = sample * p.c * 4 + due + pri + stale
        a = int(np.argmax(score))
        self.trace.append({"t": t, "action": a, "hop_conf": glob["hop_conf"],
                           "Q": sample.tolist(), "explore": [0.0] * self.n,
                           "due": due.tolist(), "priority": pri.tolist(),
                           "stale": stale.tolist(), "score": score.tolist()})
        return a


class TracedFixed(FixedSweep):
    name = "fixed"
    traced = False

    def __init__(self):
        self.trace = []

    def select(self, F, glob, t):
        self.i = (self.i + 1) % self.n
        self.trace.append({"t": t, "action": self.i, "hop_conf": 0.0,
                           "reason": "open-loop sweep: next band in sequence, blind to outcomes"})
        return self.i


class TracedRandom(RandomScan):
    name = "random"
    traced = False

    def __init__(self):
        self.trace = []

    def select(self, F, glob, t):
        a = int(self.rng.integers(self.n))
        self.trace.append({"t": t, "action": a, "hop_conf": 0.0,
                           "reason": "uniform random band choice"})
        return a


class TracedGreedy(GreedyScan):
    name = "greedy"
    traced = False

    def __init__(self, eps=0.1):
        super().__init__(eps)
        self.trace = []

    def select(self, F, glob, t):
        if self.rng.random() < self.eps:
            a = int(self.rng.integers(self.n))
            self.trace.append({"t": t, "action": a, "hop_conf": 0.0,
                               "reason": f"{self.eps:.0%} random exploration draw"})
            return a
        unseen = np.flatnonzero(F[:, 1] >= 1.0)
        if len(unseen) and F[:, 0].max() < 0.2:
            a = int(self.rng.choice(unseen))
            self.trace.append({"t": t, "action": a, "hop_conf": 0.0,
                               "reason": "coverage: never/long-unvisited band, no learned hit rate yet"})
            return a
        a = int(np.argmax(F[:, 0] + 1e-6 * self.rng.random(self.n)))
        self.trace.append({"t": t, "action": a, "hop_conf": 0.0,
                           "reason": f"greedy: best recent hit rate ({F[a, 0]:.2f})"})
        return a


def make_traced_scheduler(name, params=None):
    if name == "ucb":
        return TracedUCB(params)
    if name == "thompson":
        return TracedThompson(params)
    if name == "fixed":
        return TracedFixed()
    if name == "random":
        return TracedRandom()
    if name == "greedy":
        return TracedGreedy()
    raise ValueError(f"unknown algo {name}")


def run_episode_traced(env, scheduler, seed, rw=None, scfg=None):
    """run_episode + per-step feature snapshots + the scheduler's decision trace."""
    feats = []
    log = run_episode(env, scheduler, seed, rw, scfg,
                      on_step=lambda t, F, glob, a: feats.append(F.astype(np.float32)))
    return log, {"features": feats, "decisions": scheduler.trace}
