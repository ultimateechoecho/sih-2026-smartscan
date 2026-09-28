"""BLOCK 1b - Emitter generators.

Input : one transmitter-config row (freqs_mhz, n_freqs, pris_us, n_pris) + EnvConfig + rng
Output: Emitter(band_at[T], amp_at[T]) - which band it occupies at every slot and how strong.

Behaviour kinds (matching the PS): periodic, stagger (staggered PRI),
hopper (frequency agile), scan (spatially scanning beam).
Real TSRD values used: the emitter's frequency list -> bands, PRI ratio -> stagger ratio,
cardinality of freq/PRI lists -> behaviour kind (in "auto" mode).
Timing (period, burst, hop interval, beam period) are assumptions from EnvConfig.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig

KINDS = ("periodic", "stagger", "hopper", "scan")


@dataclass
class Emitter:
    eid: int
    kind: str
    band_at: np.ndarray          # int16[T], -1 = silent
    amp_at: np.ndarray           # float32[T]
    first_active: int
    meta: dict = field(default_factory=dict)


def freq_to_band(freqs_mhz, centres_mhz):
    f = np.atleast_1d(np.asarray(freqs_mhz, float))
    c = np.asarray(centres_mhz, float)
    return np.abs(f[:, None] - c[None, :]).argmin(axis=1)


def auto_kind(row, cfg, rng):
    if int(row.get("n_freqs", 1)) > 1:
        return "hopper"
    if int(row.get("n_pris", 1)) > 1:
        return "stagger"
    return "scan" if rng.random() < cfg.scan_overlay_prob else "periodic"


def choose_behavior_mode(stats, cfg):
    """Use 'auto' only if TSRD configs really contain >=3 behaviour types."""
    if cfg.behavior_mode != "auto":
        return cfg.behavior_mode
    kinds = {"periodic"}
    if (stats["n_freqs"] > 1).any():
        kinds.add("hopper")
    if (stats["n_pris"] > 1).any():
        kinds.add("stagger")
    if len(kinds) >= 3:
        return "auto"
    print("[b1_emitters] TSRD configs lack behaviour variety -> using behavior_mix")
    return "mix"


def pick_kind(row, cfg, rng, mode):
    if mode == "mix":
        k = list(cfg.behavior_mix.keys())
        p = np.array(list(cfg.behavior_mix.values()), float)
        return str(rng.choice(k, p=p / p.sum()))
    return auto_kind(row, cfg, rng)


def build_emitter(eid, row, centres, cfg: EnvConfig, rng, mode="mix"):
    T = cfg.episode_slots
    kind = pick_kind(row, cfg, rng, mode)
    freqs = np.atleast_1d(row.get("freqs_mhz", [])) if row is not None else np.array([])
    if len(freqs) == 0:
        freqs = np.array([rng.uniform(500, 18000)])
    bands = np.unique(freq_to_band(freqs, centres))
    band = int(rng.choice(bands))
    burst = int(rng.integers(*cfg.burst_range))
    amp0 = float(rng.uniform(*cfg.amp_range))
    start = int(rng.integers(0, int(T * cfg.start_frac_max) + 1))

    band_at = np.full(T, -1, np.int16)
    amp_at = np.full(T, -200.0, np.float32)
    meta = {}

    def mark(t, b):
        if t < T:
            e = min(t + burst, T)
            band_at[t:e] = b
            amp_at[t:e] = amp0 + rng.normal(0, 1.0)

    if kind == "periodic":
        P = int(rng.integers(*cfg.period_range))
        meta["period"] = P
        t = start
        while t < T:
            mark(t, band)
            t += P + int(rng.integers(-1, 2))

    elif kind == "stagger":
        P1 = int(rng.integers(*cfg.period_range))
        pris = np.atleast_1d(row.get("pris_us", [])) if row is not None else []
        ratio = float(np.clip(pris.max() / max(pris.min(), 1e-9), 1.4, 2.5)) \
            if len(pris) > 1 else float(rng.uniform(1.4, 2.2))
        seq = [P1, max(P1 + 8, int(P1 * ratio))]
        meta["intervals"] = seq
        t, i = start, 0
        while t < T:
            mark(t, band)
            t += seq[i % len(seq)] + int(rng.integers(-1, 2))
            i += 1

    elif kind == "hopper":
        pool = list(bands)
        while len(pool) < 3:                       # agile emitter needs >=3 channels
            pool.append(int(rng.integers(0, len(centres))))
        pool = sorted(set(pool))
        H = int(rng.integers(*cfg.hop_range))
        meta["hop_interval"] = H
        t, prev = start, -1
        while t < T:
            b = int(rng.choice([x for x in pool if x != prev]))
            mark(t, b)
            prev = b
            t += H + int(rng.integers(-2, 3))

    else:  # "scan": fast pulse train gated by a rotating antenna beam
        P = int(rng.integers(3, 8))
        t = start
        while t < T:
            mark(t, band)
            t += P
        S = int(rng.integers(*cfg.scan_period_range))
        phase = int(rng.integers(0, S))
        meta["beam_period"] = S
        env = np.abs(np.sin(np.pi * (np.arange(T) + phase) / S))
        band_at[env < 0.3] = -1
        amp_at += (20 * np.log10(np.clip(env, 0.05, 1.0))).astype(np.float32)

    if rng.random() < cfg.stop_prob:                # emitter switches off mid-mission
        stop = int(rng.integers(min(T - 1, start + 200), T + 1))
        band_at[stop:] = -1

    active = np.flatnonzero(band_at >= 0)
    if len(active) == 0:
        return None
    amp_at[band_at < 0] = -200.0
    return Emitter(eid, kind, band_at, amp_at, int(active[0]), meta)


def build_scenario(stats, centres, cfg: EnvConfig, rng):
    mode = choose_behavior_mode(stats, cfg)
    n = int(rng.integers(*cfg.n_emitters))
    out, tries = [], 0
    while len(out) < n and tries < 10 * n:
        tries += 1
        row = stats.iloc[int(rng.integers(0, len(stats)))]
        e = build_emitter(len(out), row, centres, cfg, rng, mode)
        if e is not None:
            out.append(e)
    return out


if __name__ == "__main__":
    from .b1_data import default_receiver, synthetic_stats
    cfg, rec = EnvConfig(), default_receiver()
    rng = np.random.default_rng(0)
    em = build_scenario(synthetic_stats(), rec["dwell_centres_mhz"], cfg, rng)
    for e in em:
        print(f"emitter {e.eid} {e.kind:9s} first_active={e.first_active:5d} "
              f"active_slots={(e.band_at>=0).sum():5d} meta={e.meta}")
    assert 4 <= len(em) < 9
    print("b1_emitters OK")
