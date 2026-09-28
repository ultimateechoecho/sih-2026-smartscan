"""BLOCK 2 - Receiver-constrained RF environment (gymnasium).

Input : action = index of the band (0..35) to dwell on
Output: obs = {band, hit, amplitude, t, dwell_slots}   <- ONLY what the receiver heard
        info = ground-truth flags for logging/metrics (NEVER give these to the scheduler)

Constraints modelled
  * one band per step (narrow instantaneous bandwidth)
  * dwell time per band comes from the TSRD receiver (0.05 s or 0.10 s -> 1 or 2 slots)
  * probabilistic detection that depends on amplitude vs sensitivity
  * false alarms on empty bands
  * hidden truth: emitters' band_at/amp_at arrays are private
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .b1_emitters import build_scenario
from .config import EnvConfig


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SmartScanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, receiver, stats, cfg: EnvConfig = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.centres = np.asarray(receiver["dwell_centres_mhz"], float)
        self.n_bands = len(self.centres)
        self.dwell_s = np.asarray(receiver["dwell_times_s"], float)
        self.dwell_slots = np.maximum(1, np.round(self.dwell_s / self.cfg.slot_s)).astype(int)
        self.stats = stats
        self.T = self.cfg.episode_slots
        self.action_space = spaces.Discrete(self.n_bands)
        self.observation_space = spaces.Dict({
            "band": spaces.Discrete(self.n_bands),
            "hit": spaces.Discrete(2),
            "amplitude": spaces.Box(-250.0, 0.0, shape=(1,), dtype=np.float32),
            "t": spaces.Box(0, self.T, shape=(1,), dtype=np.float32),
            "dwell_slots": spaces.Discrete(10),
        })
        self._emitters = []
        self.t = 0

    # ------------------------------------------------------------------ API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._emitters = build_scenario(self.stats, self.centres, self.cfg, self._rng)
        self.first_detect = {}
        self.t = 0
        # burst bookkeeping: a "burst" = one contiguous on-air segment of an emitter
        self._seg, self._burst_total, self._burst_seen = [], {}, set()
        for e in self._emitters:
            b = e.band_at.astype(int)
            chg = np.concatenate(([True], b[1:] != b[:-1]))
            sid = np.where(b >= 0, np.cumsum(chg) - 1, -1)
            self._seg.append(sid)
            self._burst_total[e.eid] = int(len(np.unique(sid[sid >= 0])))
        return self._obs(0, False, -150.0, 1), {"n_emitters": len(self._emitters)}

    def step(self, action):
        a, T, cfg, rng = int(action), self.T, self.cfg, self._rng
        t0 = self.t
        d = int(self.dwell_slots[a])
        t1 = min(t0 + d, T)

        active, detected_now, new_bursts = [], [], 0
        amp_max = -np.inf
        for k, e in enumerate(self._emitters):
            seg = e.band_at[t0:t1]
            m = seg == a
            if m.any():
                amp = float(e.amp_at[t0:t1][m].max())
                active.append(e.eid)
                amp_max = max(amp_max, amp)
                p = cfg.pd_max * _sigmoid((amp - cfg.sens_db) / cfg.sens_slope)
                if rng.random() < p:
                    detected_now.append(e.eid)
                    key = (e.eid, int(self._seg[k][t0:t1][m][0]))
                    if key not in self._burst_seen:
                        self._burst_seen.add(key)
                        new_bursts += 1

        true_active = len(active) > 0
        detected = len(detected_now) > 0
        false_alarm = (not true_active) and (rng.random() < cfg.pfa)
        hit = detected or false_alarm

        if detected:
            amp_obs = amp_max + rng.normal(0, 1.5)
        elif false_alarm:
            amp_obs = cfg.sens_db + rng.normal(0, 3.0)
        else:
            amp_obs = rng.normal(-110.0, 3.0)

        new = [i for i in detected_now if i not in self.first_detect]
        for i in new:
            self.first_detect[i] = t0

        self.t = t1
        pending = sum(1 for e in self._emitters
                      if e.first_active <= t0 and e.eid not in self.first_detect)
        terminated = self.t >= T
        info = {"t0": t0, "dwell_slots": d, "dwell_s": d * cfg.slot_s,
                "true_active": true_active, "detected": detected,
                "false_alarm": bool(false_alarm), "n_new": len(new),
                "n_new_burst": new_bursts,
                "n_pending": pending}
        return self._obs(a, hit, amp_obs, d), 0.0, terminated, False, info

    def _obs(self, band, hit, amp, d):
        return {"band": int(band), "hit": int(hit),
                "amplitude": np.array([amp], np.float32),
                "t": np.array([self.t], np.float32), "dwell_slots": int(d)}

    # ------------------------------------------------- EVALUATION / DEMO ONLY
    def emitter_summary(self):
        return [{"eid": e.eid, "kind": e.kind, "first_active": e.first_active,
                 "first_detect": self.first_detect.get(e.eid),
                 "active_slots": int((e.band_at >= 0).sum()), "meta": e.meta,
                 "n_bursts": self._burst_total[e.eid],
                 "n_bursts_det": sum(1 for (i, _) in self._burst_seen if i == e.eid)}
                for e in self._emitters]

    def oracle_band_active(self, band, t_lo, t_hi):
        t_lo, t_hi = max(0, int(t_lo)), min(self.T, int(t_hi))
        return any((e.band_at[t_lo:t_hi] == band).any() for e in self._emitters)

    def truth_grid(self):
        """(n_bands, T) amplitude grid, -200 = silent. For the demo heatmap only."""
        g = np.full((self.n_bands, self.T), -200.0, np.float32)
        for e in self._emitters:
            idx = np.flatnonzero(e.band_at >= 0)
            g[e.band_at[idx], idx] = np.maximum(g[e.band_at[idx], idx], e.amp_at[idx])
        return g


if __name__ == "__main__":
    from .b1_data import default_receiver, synthetic_stats
    env = SmartScanEnv(default_receiver(), synthetic_stats())
    obs, _ = env.reset(seed=1)
    assert set(obs) == {"band", "hit", "amplitude", "t", "dwell_slots"}, "truth must not leak"
    hits, steps = 0, 0
    done = False
    while not done:                                   # fixed round-robin sweep
        obs, _, done, _, info = env.step(steps % env.n_bands)
        hits += obs["hit"]
        steps += 1
    s = env.emitter_summary()
    got = sum(x["first_detect"] is not None for x in s)
    print(f"round-robin: steps={steps} hits={hits} intercepted {got}/{len(s)} emitters")
    print("b2_env OK")
