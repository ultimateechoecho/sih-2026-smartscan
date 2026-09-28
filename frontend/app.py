"""Prototype frontend.   Run from the project root:   streamlit run frontend/app.py

Tab 1  Live scan     run one scenario with a chosen scheduler, replay it step by step:
                     hidden truth heatmap (demo only) + the receiver's dwells (hit/miss)
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
from smartscan.b6_scheduler import ALGOS, load_tuned, make_scheduler
from smartscan.config import EnvConfig, RESULTS_DIR, SEED_BASE, MODELS_DIR
from smartscan.evaluate import run_benchmark, save_outputs
from smartscan.metrics import ORDER, episode_metrics
from smartscan.runner import make_env, run_episode

st.set_page_config(page_title="Smart Scan Strategy - SIH26055", layout="wide")
st.title("Smart Scan Strategy for Electronic Warfare")
st.caption("SIH26055 (DRDO) - ML-based ES receiver scheduler prototype. "
           "The scheduler sees only hit/miss + amplitude of the band it scanned.")

rec = load_receiver()
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
    log = run_episode(env, make_scheduler(algo_name, params), SEED_BASE[split] + int(idx))
    return log, env.truth_grid(), rec


def draw(log, truth, upto, show_truth, bin_slots=10):
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
    ax.legend(loc="upper right", fontsize=8)
    return fig


tab1, tab2, tab3 = st.tabs(["Live scan", "Benchmark", "Data & blocks"])

# ---------------------------------------------------------------- tab 1
with tab1:
    c1, c2 = st.columns([1, 3])
    if c1.button("Run scenario", type="primary") or "log" not in st.session_state:
        with st.spinner("simulating mission..."):
            st.session_state["log"], st.session_state["truth"], _ = run(algo, split, seed_i, cfg)
            st.session_state["algo_run"] = algo
    log, truth = st.session_state["log"], st.session_state["truth"]
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
    st.subheader(f"Episode result - {st.session_state['algo_run']}")
    k = st.columns(5)
    k[0].metric("Pd (emitters)", f"{m['pd']:.2f}")
    k[1].metric("Interception ratio", f"{m['interception_ratio']:.3f}")
    k[2].metric("Intercept delay (s)", f"{m['intercept_delay_cens_s']:.1f}")
    k[3].metric("Avg reward", f"{m['avg_reward']:.0f}")
    k[4].metric("Decision latency", f"{m['latency_ms']:.3f} ms")
    st.dataframe(pd.DataFrame(log.emitters)[["eid", "kind", "first_active", "first_detect",
                                             "n_bursts", "n_bursts_det"]])

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
        st.dataframe(df[ORDER].style.format("{:.3f}"))
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
