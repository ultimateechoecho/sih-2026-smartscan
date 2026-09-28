"""BLOCK 5 - Priority filter + reward functions.

No threat / decoy labels exist in the PS, so "interesting" is defined statistically:
    surprise = |observed_hit - expected_p| / sqrt(expected_p * (1 - expected_p))
A band that behaves as the agents predicted ("explained") gets low priority;
a band that just did something unexpected ("novel") gets high priority.
Priority never reaches a hard zero: a starvation term slowly raises stale bands.

Input : per observation (band, hit, expected_p);   per decision (visit ages, agent confs)
Output: priority score per band in ~[0, 1.3]  (feature #8 for the scheduler)

Also hosts the two reward functions:
    observable_reward  - computable online by the receiver (used by the scheduler)
    evaluation_reward  - uses simulator truth (training objective + reported metric)
"""
import numpy as np

from .config import RewardWeights


class PriorityFilter:
    def __init__(self, n_bands, decay=0.75, w_starve=0.25, surprise_cap=5.0):
        self.n, self.decay, self.w_starve, self.cap = n_bands, decay, w_starve, surprise_cap
        self.reset()

    def reset(self):
        self.s = np.zeros(self.n)

    def update(self, band, hit, expected_p):
        p = float(np.clip(expected_p, 0.02, 0.98))
        z = min(abs(hit - p) / np.sqrt(p * (1 - p)), self.cap) / self.cap    # 0..1
        self.s[band] = max(self.decay * self.s[band], z)                     # peak-hold + decay
        return z

    def scores(self, visit_age, explained_conf):
        """visit_age: 0..1 (time since last visit / cap); explained_conf: 0..1 per band."""
        return self.s * (1.0 - np.asarray(explained_conf)) + self.w_starve * np.asarray(visit_age)


def observable_reward(hit, novel, dwell_slots, rw: RewardWeights = RewardWeights()):
    """What the receiver can compute itself: hit (+ novelty after long silence), per slot."""
    return (rw.o_hit * float(hit) + rw.o_novel * float(hit and novel)) / max(1, dwell_slots)


def evaluation_reward(info, rw: RewardWeights = RewardWeights()):
    """Truth-aware reward from env.info: newly intercepted burst / emitter, miss, false alarm,
    scan cost and delay (emitters on-air but not yet intercepted). Re-detecting the SAME
    burst earns nothing, so camping on one band cannot inflate the reward."""
    r = rw.w_burst * info["n_new_burst"] + rw.w_new * info["n_new"]
    if info["false_alarm"]:
        r -= rw.w_fa
    if info["true_active"] and not info["detected"]:
        r -= rw.w_miss
    r -= rw.w_cost * info["dwell_s"]
    r -= rw.w_pending * info["n_pending"] * info["dwell_s"]
    return r


if __name__ == "__main__":
    pf = PriorityFilter(4)
    z_novel = pf.update(0, hit=1, expected_p=0.03)      # unexpected hit
    z_expl = pf.update(1, hit=1, expected_p=0.90)       # predicted hit
    z_quiet = pf.update(2, hit=0, expected_p=0.03)      # boring miss
    print("surprise novel/explained/quiet:", round(z_novel, 2), round(z_expl, 2), round(z_quiet, 2))
    assert z_novel > z_expl > 0 and z_novel > z_quiet
    sc = pf.scores(np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(4))
    print("priority:", sc.round(2))
    assert sc[0] > sc[1] and sc[3] > sc[2]              # novel first; starved band creeps up
    info = dict(detected=True, n_new=1, n_new_burst=1, false_alarm=False, true_active=True,
                dwell_s=0.05, n_pending=2)
    print("eval reward:", round(evaluation_reward(info), 3),
          "obs reward:", observable_reward(True, True, 1))
    print("b5_priority OK")
