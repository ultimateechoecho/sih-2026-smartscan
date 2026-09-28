"""STATE BUILDER - glue between env observations, Block 3 agents and Block 5 filter.

Input : update(band, t, hit, amp) after every dwell;  features(t) before every decision
Output: features -> float array (n_bands, 8);  glob -> {"hop_conf": float}

Columns
  0 hit_rate        discounted hit rate of the band
  1 visit_age       time since last visit / cap
  2 hit_age         time since last hit / cap
  3 periodic_conf   periodicity agent confidence
  4 due             confidence x closeness of NOW to the predicted next activity
  5 scan_conf       beam-scan agent confidence
  6 stagger_conf    stagger agent confidence
  7 priority        Block 5 priority score
Efficiency: agents re-run ONLY for the band that just produced a hit; everything
else is cached, so a decision costs O(n_bands) vector maths.
"""
import math
from collections import deque

import numpy as np

from .b3_agents import route_agents
from .b5_priority import PriorityFilter
from .config import StateConfig

FEATURE_NAMES = ["hit_rate", "visit_age", "hit_age", "periodic_conf",
                 "due", "scan_conf", "stagger_conf", "priority"]


class StateBuilder:
    F = len(FEATURE_NAMES)

    def __init__(self, n_bands, cfg: StateConfig = None):
        self.n = n_bands
        self.cfg = cfg or StateConfig()
        self.priority = PriorityFilter(n_bands)
        self.reset()

    def reset(self):
        n, c = self.n, self.cfg
        self.hr = np.full(n, c.hr_prior)
        self.last_visit = np.full(n, -c.stale_cap, float)
        self.last_hit = np.full(n, -1e6)
        self.hit_times = [deque(maxlen=c.hit_hist) for _ in range(n)]
        self.hit_amps = [deque(maxlen=c.hit_hist) for _ in range(n)]
        self.global_hits = deque(maxlen=60)
        self.pconf = np.zeros(n)
        self.scan_conf = np.zeros(n)
        self.stag_conf = np.zeros(n)
        self.hop_conf = np.zeros(n)
        self.best = [None] * n                 # best predictor per band
        self.explained = np.zeros(n)
        self.priority.reset()
        self.pred_log = []                     # (band, t_made, t_predicted) for accuracy metric

    # -------------------------------------------------------------- queries
    def time_since_last_hit(self, band, t):
        return t - self.last_hit[band]

    def _pred_time(self, b, t):
        """Next expected activity of band b at or after (t - small tolerance)."""
        best = self.best[b]
        if best is None:
            return None
        nxt, per = best["next_time"], best["period"]
        if per:
            if t > nxt:
                nxt = nxt + math.ceil((t - nxt) / per) * per
            return nxt, per
        return (nxt, None) if t <= nxt + 4 else None

    def _due(self, b, t):
        pt = self._pred_time(b, t)
        if pt is None:
            return 0.0
        nxt, per = pt
        d = abs(t - nxt)
        if per:
            d = min(d, abs(t - (nxt - per)))
        w = max(2.0, 0.06 * per) if per else 4.0
        return self.best[b]["conf"] * math.exp(-0.5 * (d / w) ** 2)

    def features(self, t):
        cap = self.cfg.stale_cap
        visit_age = np.clip((t - self.last_visit) / cap, 0, 1)
        hit_age = np.clip((t - self.last_hit) / cap, 0, 1)
        due = np.zeros(self.n)
        for b in np.flatnonzero(self.explained > 0):
            due[b] = self._due(b, t)
        pri = self.priority.scores(visit_age, self.explained)
        F = np.stack([self.hr, visit_age, hit_age, self.pconf, due,
                      self.scan_conf, self.stag_conf, pri], axis=1)
        return F, {"hop_conf": float(self.hop_conf.max())}

    # -------------------------------------------------------------- updates
    def update(self, band, t, hit, amp):
        c = self.cfg
        # 1) Block 5: how surprising was this outcome given what we believed?
        p_exp = max(self.hr[band], self._due(band, t) * 0.9)
        self.priority.update(band, hit, p_exp)
        # 2) bookkeeping
        self.last_visit[band] = t
        self.hr[band] += c.hr_alpha * (float(hit) - self.hr[band])
        if hit:
            self.last_hit[band] = t
            self.hit_times[band].append(t)
            self.hit_amps[band].append(amp)
            self.global_hits.append((t, band))
            self._refresh_agents(band, t)

    def _refresh_agents(self, b, t):
        outs, best = route_agents(list(self.hit_times[b]), list(self.hit_amps[b]),
                                  b, list(self.global_hits))
        self.pconf[b] = outs["periodic"]["conf"]
        self.scan_conf[b] = outs["scan"]["conf"]
        self.stag_conf[b] = outs["stagger"]["conf"]
        self.hop_conf[b] = outs["hopper"]["conf"]
        if best["agent"] is None:
            self.best[b], self.explained[b] = None, 0.0
            return
        self.best[b], self.explained[b] = best, best["conf"]
        if best["conf"] >= self.cfg.pred_log_min_conf:
            pt = self._pred_time(b, t + 1)
            if pt is not None and pt[0] > t:
                self.pred_log.append((b, t, pt[0]))


if __name__ == "__main__":
    sb = StateBuilder(6)
    F, g = sb.features(0)
    assert F.shape == (6, 8) and g["hop_conf"] == 0.0
    for k in range(8):                                  # band 2 fires every 60 slots
        sb.update(2, 100 + 60 * k, 1, -50.0)
    F, g = sb.features(100 + 60 * 7 + 61)
    print("features band 2:", dict(zip(FEATURE_NAMES, F[2].round(2))))
    assert F[2, 3] > 0.3 and F[2, 4] > 0.3              # periodic + due right now
    assert F[0, 4] == 0.0 and F[0, 1] == 1.0            # untouched band: not due, stale
    print("predictions logged:", len(sb.pred_log))
    print("state_builder OK")
