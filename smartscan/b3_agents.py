"""BLOCK 3 - Strategic agents (non-learned, physics-based advisors).

Every agent is a plain function of observed history only (no truth, no priors):
    periodicity_agent(hit_times)          -> autocorrelation (+FFT cross-check when data is rich)
    stagger_agent(hit_times)              -> alternating-interval pattern matcher
    scan_envelope_agent(hit_times, amps)  -> amplitude-envelope test (+ scipy curve_fit period)
    hopper_agent(band_hits, global_hits)  -> chance-corrected cross-band coincidence
Each returns a dict with at least {"conf": 0..1}; predictors also return
"next_time" (slot of next expected activity) and "period".
`route_agents` picks the most confident predictor for a band.
"""
import numpy as np
from scipy.optimize import curve_fit

NONE = {"conf": 0.0, "next_time": None, "period": None}


# ---------------------------------------------------------------- periodicity
def _acf_period(hit_times, min_lag, max_lag, tol):
    t0, tl = hit_times[0], hit_times[-1]
    L = int(tl - t0 + 1)
    if L < 2 * min_lag:
        return None
    x = np.zeros(L)
    x[(hit_times - t0).astype(int)] = 1.0
    x = np.convolve(x, np.ones(2 * tol + 1), mode="same")      # tolerate burst-width jitter
    n = 1 << int(np.ceil(np.log2(2 * L)))
    f = np.fft.rfft(x, n)
    r = np.fft.irfft(f * np.conj(f), n)[:L]
    hi = min(max_lag, L - 1)
    if hi <= min_lag:
        return None
    return int(np.argmax(r[min_lag:hi + 1]) + min_lag)


def _fft_period(hit_times, min_lag, max_lag):
    t0, tl = hit_times[0], hit_times[-1]
    L = int(tl - t0 + 1)
    x = np.zeros(L)
    x[(hit_times - t0).astype(int)] = 1.0
    spec = np.abs(np.fft.rfft(x - x.mean()))
    ks = np.arange(len(spec))
    ok = (ks >= 1) & (L / np.maximum(ks, 1) >= min_lag) & (L / np.maximum(ks, 1) <= max_lag)
    if not ok.any():
        return None
    k = ks[ok][np.argmax(spec[ok])]
    return L / k


def periodicity_agent(hit_times, min_lag=10, max_lag=400, tol=3, fft_switch=15):
    """Few hits -> autocorrelation only. Many hits -> autocorrelation AND FFT;
    agreement raises confidence, disagreement lowers it."""
    h = np.sort(np.asarray(hit_times, float))
    if len(h) < 3:
        return dict(NONE)
    P = _acf_period(h, min_lag, max_lag, tol)
    if P is None:
        return dict(NONE)
    # refine: consistency of every hit-to-hit gap with a multiple of P
    d = np.diff(h)
    k = np.maximum(1, np.round(d / P))
    ok = np.abs(d - k * P) <= np.maximum(tol, 0.06 * P * k)
    score = ok.mean()
    conf = score * min(1.0, (len(h) - 1) / 5.0)
    method, agree = "acf", None
    if len(h) >= fft_switch:
        Pf = _fft_period(h, min_lag, max_lag)
        agree = Pf is not None and abs(Pf - P) / P < 0.2
        conf *= 1.0 if agree else 0.6
        method = "acf+fft"
    if conf < 0.05:
        return dict(NONE)
    ang = 2 * np.pi * (h % P) / P                                   # robust phase estimate
    phase = (np.angle(np.exp(1j * ang).mean()) % (2 * np.pi)) * P / (2 * np.pi)
    nxt = h[-1] - ((h[-1] - phase) % P) + P
    return {"conf": float(conf), "period": float(P), "next_time": float(nxt),
            "method": method, "agree": agree}


# --------------------------------------------------------------------- stagger
def stagger_agent(hit_times, tol=3, max_interval=400):
    """Alternating intervals A,B,A,B (staggered PRI) look like noise to autocorrelation."""
    h = np.sort(np.asarray(hit_times, float))
    if len(h) < 6:
        return dict(NONE)
    s = np.diff(h)
    s = s[s <= max_interval]
    if len(s) < 5:
        return dict(NONE)
    med = np.median(s)
    lo, hi = s[s <= med], s[s > med]
    if len(lo) < 2 or len(hi) < 2:
        return dict(NONE)
    if hi.mean() - lo.mean() < 2 * tol or lo.std() > tol or hi.std() > tol:
        return dict(NONE)
    hi_flag = s > med
    alt = np.mean(hi_flag[1:] != hi_flag[:-1])
    conf = float(alt * min(1.0, len(s) / 8.0))
    if conf < 0.3:
        return dict(NONE)
    return {"conf": conf, "period": None, "next_time": float(h[-1] + s[-2])}


# ----------------------------------------------------------------- scan envelope
def _env_model(t, S, phi, b, g):
    return b + g * np.log10(np.abs(np.sin(np.pi * (t - phi) / S)) + 0.05)


def scan_envelope_agent(hit_times, amps, min_hits=6, fit_hits=10):
    """Antenna-beam sweeps make amplitude swing smoothly. Confidence from modulation
    depth; with enough hits curve_fit recovers the beam period for revisit prediction."""
    a = np.asarray(amps, float)
    if len(a) < min_hits:
        return dict(NONE)
    depth = a.std()
    conf = float(np.clip((depth - 3.0) / 8.0, 0.0, 1.0))
    out = {"conf": conf, "next_time": None, "period": None}
    if conf > 0.3 and len(a) >= fit_hits:
        t = np.asarray(hit_times, float)[-24:]
        y = a[-24:]
        best = None
        for S0 in (80, 140, 220, 320):
            try:
                p, _ = curve_fit(_env_model, t, y, p0=[S0, 0, y.mean(), 20],
                                 bounds=([40, -400, -120, 5], [400, 400, -20, 60]),
                                 maxfev=400)
                rmse = np.sqrt(np.mean((y - _env_model(t, *p)) ** 2))
                if best is None or rmse < best[0]:
                    best = (rmse, p)
            except (RuntimeError, ValueError):
                continue
        if best is not None and best[0] < 0.7 * depth:
            S, phi = best[1][0], best[1][1]
            k = np.ceil((t[-1] - phi) / S - 0.5)
            out.update({"period": float(S), "next_time": float(phi + (k + 0.5) * S),
                        "conf": float(conf * (1 - best[0] / (depth + 1e-9)))})
    return out


# ----------------------------------------------------------------------- hopper
def hopper_agent(band, global_hits, window=60, min_band_hits=3, min_global=8):
    """global_hits: list of (t, band). Fraction of this band's hits that coincide with a
    hit on ANOTHER band within +-window, corrected for chance coincidence."""
    if len(global_hits) < min_global:
        return {"conf": 0.0}
    g = np.asarray(global_hits, float)
    mine = g[g[:, 1] == band]
    others = g[g[:, 1] != band]
    if len(mine) < min_band_hits or len(others) < 3:
        return {"conf": 0.0}
    span = max(1.0, g[:, 0].max() - g[:, 0].min())
    p_chance = 1 - np.exp(-(len(others) / span) * 2 * window)
    coincide = np.mean([(np.abs(others[:, 0] - t) <= window).any() for t in mine[:, 0]])
    conf = float(np.clip((coincide - p_chance) / max(1e-6, 1 - p_chance), 0.0, 1.0))
    return {"conf": conf}


# ---------------------------------------------------------------------- router
def route_agents(hit_times, amps, band, global_hits, threshold=0.3):
    """Confidence arbitration (no learned router). Returns all agent outputs + the
    best predictor {'agent','conf','next_time','period'} above threshold."""
    outs = {"periodic": periodicity_agent(hit_times),
            "stagger": stagger_agent(hit_times),
            "scan": scan_envelope_agent(hit_times, amps),
            "hopper": hopper_agent(band, global_hits)}
    best = {"agent": None, "conf": 0.0, "next_time": None, "period": None}
    for name in ("periodic", "stagger", "scan"):
        o = outs[name]
        if o.get("next_time") is not None and o["conf"] > max(threshold, best["conf"]):
            best = {"agent": name, "conf": o["conf"],
                    "next_time": o["next_time"], "period": o.get("period")}
    return outs, best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # periodic: period 60, jitter, ~half the periods missed (sparse sampling)
    ht = np.array([k * 60 + rng.integers(-2, 3) for k in range(30) if rng.random() < 0.5]) + 100
    o = periodicity_agent(ht)
    print("periodic:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items()})
    assert o["conf"] > 0.4 and abs(o["period"] - 60) <= 3
    # non-periodic noise
    o2 = periodicity_agent(np.sort(rng.integers(0, 2000, 12)))
    print("noise conf:", round(o2["conf"], 2))
    assert o2["conf"] < o["conf"]
    # stagger 50/80
    seq, t = [], 100
    for i in range(14):
        seq.append(t)
        t += 50 if i % 2 == 0 else 80
    o = stagger_agent(seq)
    print("stagger:", o)
    assert o["conf"] > 0.5
    # scan envelope: smooth swings vs constant amplitude
    tt = np.arange(0, 600, 25.0)
    amp = -60 + 20 * np.log10(np.abs(np.sin(np.pi * tt / 200)) + 0.05)
    o = scan_envelope_agent(tt, amp)
    print("scan:", {k: v for k, v in o.items()})
    assert o["conf"] > 0.3
    assert scan_envelope_agent(tt, -50 + rng.normal(0, 1, len(tt)))["conf"] < 0.2
    # hopper: hits alternate across bands
    gh = [(t, int(i % 4)) for i, t in enumerate(range(0, 800, 40))]
    print("hopper:", hopper_agent(0, gh))
    print("b3_agents OK")
