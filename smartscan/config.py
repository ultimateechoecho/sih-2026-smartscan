"""Central configuration. Every block imports its constants from here.

Nothing in this file needs the dataset; it is safe to import anywhere.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "tsrd_raw"          # downloaded TSRD .h5 files
MODELS_DIR = ROOT / "models"             # tuned scheduler parameters (json)
RESULTS_DIR = ROOT / "results"           # evaluation outputs (json/csv/png)

STATS_PATH = DATA_DIR / "transmitter_stats.pkl"   # output of Block 1 extraction
RECEIVER_PATH = DATA_DIR / "receiver.json"        # 36 dwell centres + dwell times

TSRD_REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"

# Scenario splits: each split draws episode seeds from its own range, and
# transmitter configs are partitioned (see b1_data.split_stats).
SEED_BASE = {"train": 0, "val": 100_000, "test": 200_000}


@dataclass
class EnvConfig:
    """Simulator settings (Block 1 emitters + Block 2 receiver constraints)."""
    slot_s: float = 0.05                 # one time slot = 50 ms (shortest dwell in TSRD receiver)
    episode_slots: int = 4000            # 200 s mission
    n_emitters: tuple = (4, 9)           # emitters per episode, [low, high)

    # How emitter behaviour is chosen from a TSRD transmitter config:
    #  "auto": n_freqs>1 -> frequency agile, n_pris>1 -> staggered PRI, else periodic
    #          (a share of periodic ones get a beam-scan overlay)
    #  "mix" : sample kind from behavior_mix (used automatically if TSRD configs
    #          do not contain at least 3 different behaviour types)
    behavior_mode: str = "auto"
    behavior_mix: dict = field(default_factory=lambda: {
        "periodic": 0.35, "stagger": 0.20, "hopper": 0.25, "scan": 0.20})
    scan_overlay_prob: float = 0.30

    # ---- timing assumptions (NOT taken from TSRD, tune freely) ----
    period_range: tuple = (30, 160)      # slots between bursts, periodic emitters
    burst_range: tuple = (2, 8)          # burst length in slots
    hop_range: tuple = (20, 80)          # slots between hops, agile emitters
    scan_period_range: tuple = (60, 300) # antenna beam period, slots
    start_frac_max: float = 0.6          # emitters switch on within first 60% of mission
    stop_prob: float = 0.3               # chance an emitter switches off mid-mission
    amp_range: tuple = (-70.0, -35.0)    # emitter amplitude range (dB, arbitrary ref)

    # ---- receiver model ----
    sens_db: float = -85.0               # amplitude at which detection prob = 50%
    sens_slope: float = 3.0
    pd_max: float = 0.98
    pfa: float = 0.01                    # false-alarm prob per dwell with no emitter


@dataclass
class RewardWeights:
    """Evaluation reward (uses simulator truth) and observable reward (does not)."""
    # evaluation / training objective (truth-aware)
    w_burst: float = 1.0      # each newly intercepted emitter burst (counted once per burst)
    w_new: float = 5.0        # first interception of an emitter
    w_fa: float = 1.0         # false alarm
    w_miss: float = 0.2       # active band, but missed
    w_cost: float = 0.5       # per second of dwell (scan cost)
    w_pending: float = 0.5    # per second, per emitter that is on-air but not yet intercepted
    # observable reward (what the scheduler can actually compute online)
    o_hit: float = 1.0
    o_novel: float = 0.5      # bonus: hit on a band silent for a long time
    novel_gap_slots: int = 200


@dataclass
class StateConfig:
    stale_cap: int = 100      # slots at which staleness features saturate (~2.5 sweep cycles)
    hr_alpha: float = 0.2     # EMA step for per-band hit rate
    hr_prior: float = 0.03
    hit_hist: int = 48        # hit-time memory per band
    pred_log_min_conf: float = 0.4
