# Smart Scan Strategy for Electronic Warfare — Architecture Blueprint
**SIH26055 (DRDO)**

---

## Problem Statement Recap (one line)
Build an ML-based Electronic Support receiver scheduler that decides which frequency band to scan and when, learning purely from its own hit/miss feedback, with no prior intelligence of emitters — minimizing intercept time while maximizing interception rate against agile, spatially-scanning, and periodic emitters.

## Scope Note (important, stated explicitly for judges)
- **Core PS requirement:** binary transmission/non-transmission ground truth per band per time step. No threat/decoy/emitter-identity labels exist in the PS's own data definition.
- **This blueprint's extensions** (Strategic Agents, Priority Filter) go beyond literal PS minimum toward real-world operational fidelity. They are built on physics-derived statistics only (periodicity, hopping variance, amplitude envelope) — never on invented ground truth like "threat level" or "decoy" labels.

---

## Overall Pipeline

```
Block 1: Simulation (RF environment + data)
        ↓
Block 2: Model Constraints (bandwidth-limited, partial observability)
        ↓
Block 3: Strategic System (parallel lightweight physics agents)
        ↓
Block 5: Priority Filter (surprise/novelty scoring, reward shaping)
        ↓
Block 6: Main RL Scheduler (core decision engine)
        ↓
back to Block 2 (receiver executes action) → new observation → loop
```

---

## Block 1 — Simulation (Data Preparation)

**Source:** Hugging Face Alan Turing Radar Dataset (TSRD).

**Correct usage (important):** TSRD is a static deinterleaving benchmark, not an interactive environment — it has no action→consequence loop, so it cannot be the RL training environment itself.

**Plan:**
1. Extract emitter behavior statistics from a small TSRD subset (few hundred pulse trains, not full 4B pulses) — hopping patterns, inter-pulse intervals, amplitude ranges, pulse-width ranges. Use TSRD's own Python loader (from its GitHub repo).
2. Build a custom interactive simulator using `gymnasium`, parameterized by those extracted statistics. This generated environment — not the TSRD file — is what the RL agent actually trains against.
3. Reframe train/val/test as **scenario splits**, not file splits:
   - Train scenarios: emitter configs seen during training
   - Val scenarios: used for hyperparameter tuning
   - Test scenarios: entirely unseen emitter configs, used only for final demo/evaluation
4. Metrics computed via **episode rollout** (Pd, Pfa, avg intercept time, avg reward, etc.), not pointwise accuracy — matches PS's own figures-of-merit definition.
5. Latency measured directly (`time.perf_counter()` around each decision step).
6. Lightweight frontend (Streamlit/React) for live demo: band-occupancy grid, current scan highlighted, running metrics.

---

## Block 2 — Model Constraints (Realistic Receiver Limits)

**Purpose:** enforce genuine partial observability — the actual mechanic that makes the problem hard.

**Design:**
- `action_space = Discrete(N)` — pick 1 of N bands per step (k=1 receiver channel for prototype).
- Truth grid (N × T occupancy) stored privately inside env, **never exposed** to the policy.
- Observation returned to agent = only the outcome of the band it scanned (hit/miss, PDW if hit) + its own running history features — not the full truth.
- Dwell time: fixed at 1 time step per action for prototype (stretch goal: variable dwell with detection-probability curve `P(detect|active) = 1 - exp(-dwell/tau)`).
- `time_since_last_visit[band]` tracked per band — feeds both Priority Filter and reward shaping.

**Validation check:** run a hardcoded fixed round-robin policy through the env first; confirm it reproduces the PS's described open-loop failure mode before training anything. If it doesn't fail, the constraint isn't real yet.

---

## Block 3 — Strategic System (Parallel Physics Agents)

**Role:** lightweight, non-ML, physics-based detectors running parallel to the main scheduler, each specialized in one signal-behavior signature. They suggest features, they never decide actions — the RL scheduler remains the sole actor.

**Why non-ML / no prior intelligence violated:** all four agents run live statistical tests on observed history only — no pre-mission data, no emitter-specific lookup tables.

### 3.1 Periodicity Agent
- **Algorithm:** dual-method, length-gated.
  - History < ~15–20 samples → autocorrelation only (`numpy.correlate`).
  - History ≥ threshold → autocorrelation + FFT (`numpy.fft.fft`) cross-checked; agreement → high confidence; disagreement → flagged uncertain.
- **Why:** autocorrelation handles sparse/irregular early data (matches how bands are only sampled when scanned); FFT adds a fast, independent confirmation once data is richer. Directly answers the PS's explicit "intercept a periodic scan receiver" requirement.

### 3.2 Hopper Agent
- **Algorithm:** frequency-sequence variance/entropy + switch-rate counter.
- **Why:** hopping = high frequency variance, short per-band dwell — simple arithmetic test, no training needed. Matches PS's "frequency agile emitter" requirement.

### 3.3 Scan-Envelope Agent
- **Algorithm:** amplitude-vs-time curve fit (`scipy.optimize.curve_fit`) or peak detection on amplitude series.
- **Why:** antenna beam sweep produces a physically smooth rise-fall amplitude pattern — deterministic physics test. Matches PS's "spatially scanning emitter" requirement.

### 3.4 Stagger Agent
- **Algorithm:** pattern-match on TOA-interval sequence against small repeating sets (2-value or 3-value alternation).
- **Why:** staggered-PRI intervals alternate rather than stay constant — autocorrelation/FFT would misread this as noise. Needs a dedicated interval-pattern matcher. Kept as a separate agent from Periodicity (different physics signature, would blur detection if merged).

### Routing / Arbitration
- **Logic:** confidence-threshold argmax (not a learned router — avoids unjustified extra training burden).
```
for band with sufficient history:
    scores = {agent: agent.confidence(band_history) for agent in [periodicity, hopper, scan, stagger]}
    best_agent, best_score = argmax(scores)
    if best_score > threshold:
        emit (band, predicted_next_active_time, confidence) as feature
    else:
        emit "no strong pattern" flag
```
- Output feeds the scheduler's state vector as extra features — never overrides the scheduler's action.

---

## Block 5 — Priority Filter (Surprise-Based Prioritization)

**Reframe (important):** not "suspicious vs normal" (no threat ground truth exists) — instead **explained vs novel**, based on prediction-error statistics.

**Math — surprise score (Z-score style):**
```
surprise = |observed_value - predicted_value| / σ
```
`predicted_value` reuses Block 3 agent outputs directly — no new prediction module.

- Low surprise + high agent confidence + stable history → explained band, low priority (but never zero — small residual weight avoids permanent blind spots).
- High surprise (new emitter, sudden amplitude jump, broken pattern) → high priority.

**Priority score:**
```
priority_score(band) = f(surprise, agent_confidence, time_since_last_visit)
```
`time_since_last_visit` rising slowly increases priority — prevents starvation of any band.

**Usage:** feeds (a) scheduler state as extra feature, (b) reward shaping (small bonus for catching high-surprise hits, small cost reduction for correctly deprioritizing stable bands). Soft weighting only — never a hard skip, since a "boring" band could still surprise later.

---

## Block 6 — Main RL Scheduler (Core Engine)

**State vector per band** (concatenated across all N bands):
```
state[band] = [
    hit/miss history (last k steps),          # Block 1/2
    time_since_last_visit,                    # Block 2
    periodicity_agent: predicted_time, conf,  # Block 3
    hopper_agent: confidence,                 # Block 3
    scan_agent: confidence,                   # Block 3
    stagger_agent: confidence,                # Block 3
    priority_score                            # Block 5
]
```

**Core algorithm (finalized): Contextual Multi-Armed Bandit — UCB / Thompson Sampling**
- Each band = an arm; state features = context.
- **Why chosen:** built-in, principled explore/exploit math (UCB confidence bound or Thompson posterior sampling); fast to train (no deep net, no replay buffer); fully explainable to judges ("this band chosen because its upper confidence bound was highest").
- **Stretch/comparison path:** DQN, if time permits — can capture multi-step consequences a bandit's near-myopic view misses. Kept as an upgrade demonstration, not the core dependency.
- **PPO explicitly not used** — only justified for continuous/variable dwell-time action spaces, which are out of scope for the prototype.

**Time-efficiency design:**
- Incremental state updates: only the just-scanned band's features refresh each step; all others carry forward with incremented `time_since_last_visit` — O(1) per step, not O(N).
- Agent scores (Block 3) cached, recomputed only when a band receives a new observation.
- Fixed-size, small state (7–8 features × N bands) keeps latency low.

**2D (time × frequency) prediction — decomposition, not a direct 2D forecast:**
- **Frequency axis:** handled directly by the discrete action space (`Discrete(N)`), chosen by the bandit.
- **Time axis:** handled implicitly via the Periodicity Agent's `predicted_next_time` feature inside state — the bandit learns to time its band choice using that feature, rather than needing a separate 2D grid-forecasting model (avoids unnecessary complexity like CNN-over-grid approaches, which don't fit the sparse, sequential, one-band-per-step action structure).

---

## Evaluation (all blocks, tied together)

Run trained scheduler through held-out **test scenarios** (unseen emitter configs) via full-episode rollout. Compute, per PS's own figures of merit:
- Probability of Detection
- Probability of False Alarm
- Sensitivity
- Average Intercept Rate
- Average Reward / Cost function
- Percentage of correct predictions
- Average Intercept Time Error

Compare against baselines: fixed sequential sweep, random scan, greedy revisit-last-active — this comparison ladder is the core empirical demonstration of the entire solution's value.

---

## Summary Table

| Block | Core Method | Training Needed? |
|---|---|---|
| 1. Simulation | Custom `gymnasium` env, seeded by TSRD stats | No (data generation only) |
| 2. Constraints | Partial-observability design (hidden truth grid) | No |
| 3. Strategic Agents | Autocorrelation, FFT, variance/entropy, curve-fit, pattern-match | No |
| 5. Priority Filter | Z-score surprise scoring | No |
| 6. Main Scheduler | UCB / Thompson Sampling bandit (DQN stretch) | Yes — this is the one trained component |

Only Block 6 requires actual model training; every other block is deterministic/statistical, keeping the "no prior intelligence" claim honest and the system fully explainable.
