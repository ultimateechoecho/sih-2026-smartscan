"""Prototype frontend.   Run from the project root:   streamlit run frontend/app.py

Tab 1  Live scan & decision board
                     run one scenario with a chosen scheduler, replay it step by step:
                     Frequency x Time scan map (hidden truth overlay demo-only) + dwells,
                     all PS figures of merit (Pd, Pfa, sensitivity, intercept rate, reward,
                     % correct predictions, intercept time error) and a live decision board:
                     current + next scan band, hit/miss feedback, decision reason,
                     detected signal behaviour + confidence, per-band priority.
Tab 2  Benchmark     paired comparison fixed / random / greedy / UCB / Thompson on the
                     held-out split with every PS figure of merit
Tab 3  Data & blocks receiver bands from TSRD, transmitter-config stats, block I/O map
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from smartscan.b1_data import load_receiver, load_stats, split_stats
from smartscan.b6_scheduler import ALGOS, load_tuned
from smartscan.config import EnvConfig, RESULTS_DIR, SEED_BASE, MODELS_DIR
from smartscan.evaluate import run_benchmark, save_outputs
from smartscan.metrics import ORDER, episode_metrics
from smartscan.runner import make_env
from server.trace import make_traced_scheduler, run_episode_traced

st.set_page_config(page_title="Smart Scan Strategy - SIH26055", layout="wide")
st.title("Smart Scan Strategy for Electronic Warfare")
st.caption("SIH26055 (DRDO) - ML-based ES receiver scheduler prototype. "
           "The scheduler sees only hit/miss + amplitude of the band it scanned.")

rec = load_receiver()
CENTRES = list(rec["dwell_centres_mhz"])
tuned = {a: (MODELS_DIR / f"best_{a}.json").exists() for a in ("ucb", "thompson")}

with st.sidebar:
    st.header("Scenario")
    split = st.selectbox("Split (unseen emitter configs)", ["test", "val", "train"])
    seed_i = st.number_input("Episode index", 0, 9999, 0)
    n_em = st.slider("Emitters (min, max)", 2, 14, (4, 9))
    st.header("Scheduler")
    algo = st.selectbox("Algorithm", list(ALGOS), index=3)
    st.write("Tuned params:", "yes" if tuned.get(algo) else "defaults (run train.py)")
    show_truth = st.checkbox("Overlay hidden truth (demo only)", value=True)

cfg = EnvConfig(n_emitters=(n_em[0], max(n_em[0] + 1, n_em[1])))


@st.cache_resource
def get_env(split, n_lo, n_hi):
    return make_env(split, EnvConfig(n_emitters=(n_lo, n_hi)))


def run(algo_name, split, idx, cfg_):
    env = get_env(split, *cfg_.n_emitters)
    params = load_tuned(algo_name) if algo_name in ("ucb", "thompson") else None
    log, trace = run_episode_traced(env, make_traced_scheduler(algo_name, params),
                                    SEED_BASE[split] + int(idx))
    return log, trace, env.truth_grid()


def _fmt(v, spec):
    """NaN-safe metric text."""
    return "—" if v != v else spec.format(float(v))


# agent confidence columns in the state-builder feature matrix
BEHAVIOUR_COLS = (("Periodic", 3), ("Scan", 5), ("Stagger", 6))


def behaviour_of(F, b, hop_conf_g, thresh=0.30):
    """Detected behaviour label + confidence for band b (agents advise, truth unseen)."""
    cands = [(lbl, float(F[b, c])) for lbl, c in BEHAVIOUR_COLS] + [("Hopping", float(hop_conf_g))]
    lbl, conf = max(cands, key=lambda x: x[1])
    return (lbl if conf >= thresh else "Unknown"), conf


def draw(log, truth, upto, show_truth, bin_slots=10):
    """Frequency x Time scan map: where activity occurs (truth, optional) and
    where the receiver actually scanned (dwells, hit/miss)."""
    n_b, T = truth.shape
    fig, ax = plt.subplots(figsize=(11, 5))
    if show_truth:
        nb = T // bin_slots
        g = truth[:, :nb * bin_slots].reshape(n_b, nb, bin_slots).max(axis=2)
        ax.imshow(np.where(g > -199, 1.0, 0.0), aspect="auto", cmap="Greys", alpha=0.35,
                  extent=[0, T, n_b - 0.5, -0.5], interpolation="nearest")
    t = np.array(log.t[:upto]); b = np.array(log.band[:upto]); h = np.array(log.hit[:upto], bool)
    ax.scatter(t[~h], b[~h], s=6, c="#9bb", marker="|", label="dwell: miss")
    ax.scatter(t[h], b[h], s=28, c="#d62728", marker="o", label="dwell: HIT")
    if upto:
        ax.axvline(t[-1], color="#1f77b4", lw=1)
    ax.set_xlim(0, T); ax.set_ylim(n_b - 0.5, -0.5)
    ax.set_xlabel(f"time slot ({log.slot_s*1000:.0f} ms each)"); ax.set_ylabel("band index (36 dwell centres)")
    ax.set_title("Frequency × Time scan map (grey = signal activity, dots = receiver dwells)",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    return fig


def priority_chart(F, cur, nxt, n_b):
    """Band priority (Block 5 score) for all bands; current/next highlighted."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = ["#1f77b4"] * n_b
    if cur is not None:
        colors[cur] = "#d62728"
    if nxt is not None:
        colors[nxt] = "#ff7f0e"
    ax.bar(range(n_b), F[:, 7], color=colors)
    ax.set_xlabel("band index"); ax.set_ylabel("priority score")
    ax.set_ylim(0, max(0.05, float(F[:, 7].max()) * 1.2))
    ax.set_title("Band priority (novelty + behaviour + revisit) — red = current, orange = next",
                 fontsize=9)
    return fig


def top_bands_table(F, hop_g, cur, nxt, k=8):
    rows = []
    for b in np.argsort(-F[:, 7])[:k]:
        beh, conf = behaviour_of(F, int(b), hop_g)
        rows.append({
            "band": int(b),
            "centre_MHz": round(CENTRES[int(b)], 1),
            "status": "NOW" if b == cur else ("NEXT" if b == nxt else ""),
            "behaviour": beh,
            "conf": round(conf, 2),
            "hit_rate": round(float(F[b, 0]), 3),
            "visit_age": round(float(F[b, 1]), 2),
            "due": round(float(F[b, 4]), 2),
            "priority": round(float(F[b, 7]), 3),
        })
    return pd.DataFrame(rows)


tab1, tab2, tab3 = st.tabs(["Live scan & decision board", "Benchmark", "Data & blocks"])

# ---------------------------------------------------------------- tab 1
with tab1:
    c1, c2 = st.columns([1, 3])
    if c1.button("Run scenario", type="primary") or "log" not in st.session_state:
        with st.spinner("simulating mission..."):
            st.session_state["log"], st.session_state["trace"], st.session_state["truth"] = \
                run(algo, split, seed_i, cfg)
            st.session_state["algo_run"] = algo
    log = st.session_state["log"]
    trace = st.session_state["trace"]
    truth = st.session_state["truth"]
    feats, decisions = trace["features"], trace["decisions"]
    n_steps = len(log.t)
    step = c1.slider("Replay position (dwells)", 1, n_steps, n_steps)
    play = c1.button("Play")
    fig_slot = c2.empty()
    fig_slot.pyplot(draw(log, truth, step, show_truth))
    if play:
        for s in range(1, n_steps, max(1, n_steps // 60)):
            fig_slot.pyplot(draw(log, truth, s, show_truth))
            time.sleep(0.05)
        fig_slot.pyplot(draw(log, truth, n_steps, show_truth))

    m = episode_metrics(log)
    st.subheader(f"Figures of merit — {st.session_state['algo_run']}")
    r1 = st.columns(4)
    r1[0].metric("Pd — Probability of Detection", _fmt(m["pd"], "{:.0%}"))
    r1[1].metric("Pfa — Probability of False Alarm", _fmt(m["pfa"], "{:.1%}"))
    r1[2].metric("Sensitivity", _fmt(m["sensitivity"], "{:.0%}"))
    r1[3].metric("Avg Intercept Rate (per 10 s)", _fmt(m["intercept_rate_10s"], "{:.2f}"))
    r2 = st.columns(4)
    r2[0].metric("Avg Intercept Delay (s)", _fmt(m["intercept_delay_cens_s"], "{:.1f}"))
    r2[1].metric("Avg Reward (efficiency)", _fmt(m["avg_reward"], "{:.0f}"))
    r2[2].metric("% Correct Predictions", _fmt(m["pct_correct_pred"], "{:.0%}"))
    r2[3].metric("Avg Intercept Time Error (s)", _fmt(m["intercept_time_err_s"], "{:.2f}"))
    st.caption(f"decision latency {_fmt(m['latency_ms'], '{:.3f}')} ms · {m['n_dwells']} dwells · "
               "delay is censored (un-intercepted emitters count to mission end)")

    # ------------------------------------------------ live decision board
    st.subheader("Live decision board")
    k = step - 1                                   # current decision (0-based)
    F_now = feats[min(step, n_steps - 1)]          # features known after `step` dwells
    hop_g = decisions[min(step, n_steps - 1)]["hop_conf"]
    cur = int(log.band[k])
    nxt = int(log.band[step]) if step < n_steps else None
    hit, det, fa = bool(log.hit[k]), bool(log.detected[k]), bool(log.fa[k])
    beh_cur, conf_cur = behaviour_of(F_now, cur, hop_g)

    b1, b2, b3 = st.columns(3)
    b1.metric("Current scan band", f"Band {cur}", f"{CENTRES[cur]:.0f} MHz")
    b1.caption(f"Signal behaviour: **{beh_cur}** ({conf_cur:.0%} confidence)"
               + (f" · hopper suspicion {hop_g:.0%}" if hop_g > 0.05 else ""))
    if fa:
        b2.error("FALSE ALARM — noise crossed the detection threshold")
    elif det:
        b2.success("HIT — active transmission detected")
    else:
        b2.info("MISS — band silent or below sensitivity")
    b2.caption("Hit / miss feedback just applied to the scheduler")
    if nxt is not None:
        b3.metric("Next scan band", f"Band {nxt}", f"{CENTRES[nxt]:.0f} MHz")
        b3.caption(f"next reason: {decisions[step]['reason']}")
    else:
        b3.metric("Next scan band", "—", "mission complete")
    b3.caption(f"current reason: **{decisions[k].get('reason', 'n/a')}**")

    p1, p2 = st.columns(2)
    with p1:
        st.pyplot(priority_chart(F_now, cur, nxt, len(CENTRES)))
    with p2:
        st.markdown("**Top bands by priority — detected behaviour & confidence**")
        st.dataframe(top_bands_table(F_now, hop_g, cur, nxt), height=290)

    st.subheader(f"Episode result — {st.session_state['algo_run']}")
    st.dataframe(pd.DataFrame(log.emitters)[["eid", "kind", "first_active", "first_detect",
                                             "n_bursts", "n_bursts_det"]])
    st.caption("emitter kinds are hidden ground truth (demo overlay only)")

# ---------------------------------------------------------------- tab 2
with tab2:
    eps = st.slider("Episodes per algorithm", 3, 40, 10)
    cached = RESULTS_DIR / f"metrics_{split}.csv"
    if st.button("Run benchmark (paired seeds)"):
        with st.spinner("running all schedulers..."):
            table, _ = run_benchmark(list(ALGOS), split, eps, jobs=1)
            st.session_state["bench"] = save_outputs(table, split, eps)
    df = st.session_state.get("bench")
    if df is None and cached.exists():
        df = pd.read_csv(cached, index_col=0)
        st.info(f"showing saved results from {cached.name}")
    if df is not None:
        cols = [c for c in ORDER if c in df.columns]     # cached files may predate new metrics
        st.dataframe(df[cols].style.format("{:.3f}"))
        fig, axs = plt.subplots(1, 4, figsize=(14, 3))
        for ax, (k, t) in zip(axs, [("pd", "Pd"), ("interception_ratio", "Interception ratio"),
                                    ("intercept_delay_cens_s", "Delay (s) lower=better"),
                                    ("avg_reward", "Avg reward")]):
            ax.bar(df.index, df[k], color=["#999" if a in ("fixed", "random", "greedy") else "#1f77b4"
                                           for a in df.index])
            ax.set_title(t, fontsize=9); ax.tick_params(axis="x", rotation=30, labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
    st.caption("Primary metrics: Pd, interception ratio, intercept delay, reward. "
               "Grey = baselines, blue = proposed bandit schedulers.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Receiver (from TSRD scan files)")
    st.caption(f"source: {rec.get('source')}")
    st.dataframe(pd.DataFrame({"band": range(len(rec["dwell_centres_mhz"])),
                               "centre_MHz": rec["dwell_centres_mhz"],
                               "dwell_s": rec["dwell_times_s"]}), height=220)
    stats = load_stats()
    parts = split_stats(stats)
    st.write({k: len(v) for k, v in parts.items()}, "transmitter configs per split")
    st.subheader("Block map")
    st.markdown("""
| Block | Module | Input -> Output |
|---|---|---|
| 1 Simulation | `b1_data`, `b1_emitters` | TSRD h5 -> receiver.json, transmitter_stats.pkl -> emitter timelines |
| 2 Constraints | `b2_env` | band index -> {hit, amplitude} only (truth hidden) |
| 3 Agents | `b3_agents` | hit times/amps -> period, next time, confidences |
| 5 Priority | `b5_priority` | outcome vs expectation -> surprise/priority score |
| 6 Scheduler | `b6_scheduler` | 36x8 features -> next band |
""")
