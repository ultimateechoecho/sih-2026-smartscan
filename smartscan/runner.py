"""RUNNER - runs one full episode; used by train, evaluate and the frontend.

Loop per dwell:  features -> scheduler.select -> env.step -> observable reward
                 -> scheduler.update -> state_builder.update -> evaluation reward -> log
The scheduler only ever sees StateBuilder features (built from hit/miss + amplitude);
`info` truth flags go to the log and to the evaluation reward, never to the scheduler.
"""
import time

from .b1_data import load_receiver, load_stats, split_stats
from .b2_env import SmartScanEnv
from .b5_priority import evaluation_reward, observable_reward
from .b6_scheduler import make_scheduler
from .config import EnvConfig, RewardWeights, SEED_BASE, StateConfig
from .metrics import EpisodeLog
from .state_builder import StateBuilder

import time as _time


def make_env(split="test", cfg: EnvConfig = None, receiver=None, stats=None):
    """Env whose emitters come from the transmitter configs of the given split."""
    receiver = receiver or load_receiver()
    stats = split_stats(stats if stats is not None else load_stats())[split]
    return SmartScanEnv(receiver, stats, cfg or EnvConfig())


def run_episode(env, scheduler, seed, rw: RewardWeights = None, scfg: StateConfig = None,
                pred_tol=4, pred_err_tol=60, on_step=None):
    """pred_tol: +-slots for counting a prediction "correct" (pct_correct_pred).
    pred_err_tol: +-slots of the wider window in which the actual next activity must
    fall for its |actual - predicted| timing error to be recorded (intercept-time-error)."""
    rw = rw or RewardWeights()
    scfg = scfg or StateConfig()
    obs, _ = env.reset(seed=seed)
    sb = StateBuilder(env.n_bands, scfg)
    scheduler.reset(env.n_bands, env.dwell_slots, seed=seed + 12345)
    log = EpisodeLog(algo=scheduler.name, seed=seed, T=env.T, slot_s=env.cfg.slot_s)
    t, done = 0, False
    while not done:
        t_dec = time.perf_counter()
        F, glob = sb.features(t)
        a = scheduler.select(F, glob, t)
        if on_step is not None:                      # optional trace hook (server/demo use)
            on_step(t, F, glob, a)
        lat = (time.perf_counter() - t_dec) * 1e3

        obs, _, done, _, info = env.step(a)
        hit, amp, d = obs["hit"], float(obs["amplitude"][0]), obs["dwell_slots"]

        novel = bool(hit) and sb.time_since_last_hit(a, t) > rw.novel_gap_slots
        scheduler.update(a, hit, observable_reward(hit, novel, d, rw))
        sb.update(a, t, hit, amp)

        log.t.append(t); log.band.append(a); log.hit.append(hit)
        log.detected.append(info["detected"]); log.active.append(info["true_active"])
        log.fa.append(info["false_alarm"]); log.dwell.append(d)
        log.r_eval.append(evaluation_reward(info, rw)); log.latency_ms.append(lat)
        t = env.t

    log.emitters = env.emitter_summary()
    for b, t_made, t_pred in sb.pred_log:                 # truth used here only for metrics
        if t_pred < env.T:
            if env.oracle_band_active(b, t_pred - pred_tol, t_pred + pred_tol + 1):
                log.pred_checks.append(True)
            t_act = env.oracle_first_active(b, max(t_made + 1, t_pred - pred_err_tol),
                                            t_pred + pred_err_tol + 1)
            if t_act is not None:
                log.pred_errors.append(abs(t_act - t_pred))
    return log


if __name__ == "__main__":
    from .metrics import episode_metrics
    env = make_env("test")
    for name in ("fixed", "ucb"):
        t0 = _time.time()
        log = run_episode(env, make_scheduler(name), seed=SEED_BASE["test"])
        m = episode_metrics(log)
        print(f"{name:6s} {_time.time()-t0:4.1f}s pd={m['pd']:.2f} delay={m['intercept_delay_cens_s']:.1f}s "
              f"reward={m['avg_reward']:.1f} dwells={m['n_dwells']} lat={m['latency_ms']:.3f}ms")
    print("runner OK")
