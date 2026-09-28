"""METRICS - the PS figures of merit, computed from an EpisodeLog.

Definitions (state these in the submission):
  pd                   emitter-level Probability of Detection = intercepted emitters / emitters
  interception_ratio   intercepted emitter bursts / all emitter bursts (PS: "interception ratio")
  sensitivity          dwell-level detection prob given an emitter was active (TP / (TP+FN))
  pfa                  false alarms / dwells on bands with no active emitter
  intercept_rate_10s   intercepted emitters per 10 s of mission
  avg_reward           total evaluation reward per episode (truth-aware)
  pct_correct_pred     share of periodicity/stagger/beam predictions confirmed by truth (+-tol slots)
  intercept_delay_s    mean time from an emitter's first activity to its interception
  intercept_delay_cens_s  same, but un-intercepted emitters count as (mission_end - first_active)
Primary metrics for ranking: pd, interception_ratio, intercept_delay_cens_s, avg_reward.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeLog:
    algo: str = ""
    seed: int = 0
    T: int = 0
    slot_s: float = 0.05
    t: list = field(default_factory=list)
    band: list = field(default_factory=list)
    hit: list = field(default_factory=list)
    detected: list = field(default_factory=list)
    active: list = field(default_factory=list)
    fa: list = field(default_factory=list)
    dwell: list = field(default_factory=list)
    r_eval: list = field(default_factory=list)
    latency_ms: list = field(default_factory=list)
    emitters: list = field(default_factory=list)
    pred_checks: list = field(default_factory=list)


def episode_metrics(log: EpisodeLog):
    det = np.asarray(log.detected, bool)
    act = np.asarray(log.active, bool)
    fa = np.asarray(log.fa, bool)
    tp, fn = int((det & act).sum()), int((~det & act).sum())
    n_inactive = int((~act).sum())

    em = log.emitters
    n_em = len(em)
    got = [e for e in em if e["first_detect"] is not None]
    delays = [(e["first_detect"] - e["first_active"]) * log.slot_s for e in got]
    cens = [((e["first_detect"] if e["first_detect"] is not None else log.T) - e["first_active"])
            * log.slot_s for e in em]
    dur = log.T * log.slot_s
    return {
        "pd": len(got) / n_em if n_em else np.nan,
        "interception_ratio": (sum(e["n_bursts_det"] for e in em) / max(1, sum(e["n_bursts"] for e in em)))
        if em else np.nan,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "pfa": fa.sum() / n_inactive if n_inactive else np.nan,
        "intercept_rate_10s": len(got) / dur * 10.0,
        "avg_reward": float(np.sum(log.r_eval)),
        "pct_correct_pred": float(np.mean(log.pred_checks)) if log.pred_checks else np.nan,
        "intercept_delay_s": float(np.mean(delays)) if delays else np.nan,
        "intercept_delay_cens_s": float(np.mean(cens)) if cens else np.nan,
        "n_dwells": len(log.t),
        "latency_ms": float(np.mean(log.latency_ms)) if log.latency_ms else np.nan,
    }


def aggregate(rows):
    """list of episode_metrics dicts -> {metric: (mean, std)} ignoring NaN."""
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], float)
        v = v[~np.isnan(v)]
        out[k] = (float(v.mean()), float(v.std())) if len(v) else (np.nan, np.nan)
    return out


PRIMARY = ["pd", "interception_ratio", "intercept_delay_cens_s", "avg_reward"]
ORDER = ["pd", "interception_ratio", "sensitivity", "pfa", "intercept_rate_10s", "avg_reward",
         "pct_correct_pred", "intercept_delay_s", "intercept_delay_cens_s", "latency_ms"]


if __name__ == "__main__":
    log = EpisodeLog(algo="demo", T=100, slot_s=0.05)
    log.detected, log.active = [True, False, False, True], [True, True, False, False]
    log.fa = [False, False, True, False]
    log.t = [0, 1, 2, 3]
    log.emitters = [{"first_active": 0, "first_detect": 10, "n_bursts": 4, "n_bursts_det": 2},
                    {"first_active": 5, "first_detect": None, "n_bursts": 6, "n_bursts_det": 0}]
    log.pred_checks = [True, False]
    m = episode_metrics(log)
    assert abs(m["interception_ratio"] - 0.2) < 1e-9 and m["pd"] == 0.5 and m["sensitivity"] == 0.5 and m["pfa"] == 0.5
    assert abs(m["intercept_delay_s"] - 0.5) < 1e-9 and m["pct_correct_pred"] == 0.5
    print(m)
    print("metrics OK")
