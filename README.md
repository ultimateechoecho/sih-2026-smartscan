


# Smart Scan Strategy for EW - SIH26055 prototype

Modular Python prototype: TSRD data -> simulated ES receiver -> physics agents ->
priority filter -> bandit scheduler -> metrics -> Streamlit demo.
Every block is a separate module with its own self-test.

## 0. Quickstart — end-to-end demo (React mission control)
```
# 1) python env
python -m venv venv
venv\Scripts\activate                 # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 2) (optional) real TSRD data -> better emitter statistics; synthetic fallback works too
python -m smartscan.b1_data --download 6 && python -m smartscan.b1_data --extract

# 3) build the React frontend (Node 18+ required)
cd frontend-react && npm install && npm run build && cd ..

# 4) run everything from one port
venv\Scripts\python -m uvicorn server.api:app --port 8000     # Linux/Mac: python3 -m ...
# -> open http://127.0.0.1:8000
#    tabs: Mission Control / Waveform Lab / Decision Inspector / Benchmark
```
Details: numbered steps 1-5 below and the "React mission-control frontend" section.
The original Streamlit UI also still runs with `streamlit run frontend/app.py`.

## 1. Setup (Python 3.10+)
```
python -m venv venv
venv\Scripts\activate          # Windows      (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python -m smartscan.selftest   # 9 x PASS = all blocks work
```
No dataset needed for this step: without TSRD files the code uses an assumed
36-band receiver and synthetic transmitter configs (prints a WARNING).

## 2. Get real TSRD data (Block 1)
```
python -m smartscan.b1_data --list          # shows scan/train_scan/*.h5 on the Hub
python -m smartscan.b1_data --download 6     # 6 small scan config files -> data/tsrd_raw/
python -m smartscan.b1_data --inspect data/tsrd_raw/scan/train_scan/config_0.h5
python -m smartscan.b1_data --extract       # -> data/receiver.json + data/transmitter_stats.pkl
python -m smartscan.b1_data                 # prints bands, rows, split sizes
```
Repo id: `alan-turing-institute/turing-synthetic-radar-dataset` (set in `config.py`).
Download 6-10 files, not the whole dataset. The extraction reads the paths verified on
`config_0.h5`: `metadata/receiver/*`, `metadata/transmitters/transmitters_X/{frequency_config/freqs_mhz,
pri_config/pris_us, pulse_width_config/pws_us, scan_config}`. If `--extract` raises KeyError,
run `--inspect` and adjust the key names in `extract_transmitters()`.
Train/val/test = transmitter configs split by source file (needs >=3 files) + separate seed ranges.

## 3. Train + validate (Block 6 tuning)
```
python -m smartscan.train --algo ucb      --quick                       # 1-2 min smoke test
python -m smartscan.train --algo ucb      --trials 24 --train-eps 5 --val-eps 8 --jobs 4
python -m smartscan.train --algo thompson --trials 24 --train-eps 5 --val-eps 8 --jobs 4
```
Random search over scheduler weights on TRAIN scenarios, best 3 re-scored on unseen VALIDATION
scenarios, winner saved to `models/best_<algo>.json`. Set `--jobs` to your core count.

## 4. Test on held-out scenarios
```
python -m smartscan.evaluate --split test --episodes 30 --jobs 4
```
Paired seeds, all 5 schedulers, writes `results/metrics_test.{json,csv}` and `compare_test.png`.

## 5. Prototype frontend
```
streamlit run frontend/app.py
```
Tab 1 replay one mission (hidden truth overlay optional), Tab 2 benchmark table/charts,
Tab 3 receiver bands, split sizes, block I/O map.

## Block map
| Block | File | Input -> Output |
|---|---|---|
| 1 data | `b1_data.py` | TSRD h5 -> receiver.json, transmitter_stats.pkl |
| 1 emitters | `b1_emitters.py` | config row -> band_at[T], amp_at[T] per emitter |
| 2 env | `b2_env.py` | band index -> {hit, amplitude, t}; truth hidden |
| 3 agents | `b3_agents.py` | hit times/amps -> period, next_time, confidence |
| 5 priority | `b5_priority.py` | outcome vs expectation -> priority; reward functions |
| state | `state_builder.py` | observations -> 36x8 feature matrix (cached) |
| 6 scheduler | `b6_scheduler.py` | features -> next band (UCB, Thompson, baselines) |
| glue | `runner.py`, `metrics.py`, `train.py`, `evaluate.py` | episode loop, PS metrics |

## Metric definitions
pd = emitters intercepted / emitters; interception_ratio = emitter bursts intercepted / bursts;
sensitivity = detected / active dwells; pfa = false alarms / empty dwells; intercept_delay =
first activity -> first interception (censored version counts misses to mission end);
pct_correct_pred = agent predictions confirmed by truth (+-4 slots).

## Rules the code enforces
* `info[...]` truth flags never reach the scheduler; emitter `labels` from TSRD are never used.
* Observable reward (online) vs evaluation reward (truth-aware, training objective/metric) are separate.
* Newly intercepted bursts are rewarded once, so camping on one band cannot inflate reward.

## Known limits (be upfront with judges)
* Timing of emitters (period 30-160 slots, bursts, hop interval, beam period) are assumptions in
  `config.EnvConfig`; TSRD supplies frequencies, behaviour type by list sizes, and PRI ratio.
* Dwell times beyond the first four bands are assumed 0.05 s in fallback mode; real ones come from h5.
* In sandbox runs (synthetic configs, tiny training) the tuned bandit only tied a fixed sweep on
  Pd/delay, improved prediction accuracy, and a greedy tracker won on reward. Re-run full training
  on real TSRD configs before claiming any improvement.
* Frequency-hopper agent is a heuristic; stagger agent needs many hits to fire.
* DQN/PPO are not implemented (bandit is the finalized core).

## React mission-control frontend (new — `server/` + `frontend-react/`)

A second, demo-oriented frontend in React (dark "EW command center" theme). The original
Streamlit app (`frontend/app.py`) is unchanged and both run off the same `smartscan` engine.

One-time setup:
    pip install fastapi            # already in requirements.txt
    cd frontend-react && npm install && npm run build

Run (single port — serves the built React app and the API):
    venv/Scripts/python -m uvicorn server.api:app --port 8000
    -> open http://127.0.0.1:8000

For React hot-reload during development: run the command above plus
`cd frontend-react && npm run dev` (Vite proxies /api to port 8000).

Tabs:
  1. Mission Control     figures-of-merit board (Pd, Pfa, sensitivity, interception ratio,
                         intercept delay/rate, reward, prediction accuracy, latency, dwells)
                         with per-metric deltas vs the fixed-sweep baseline on the same seed;
                         band×time heatmap + scan trajectory; per-emitter intercepts.
  2. Waveform Lab        the bandwidth hit/miss visualizer: live 36-band spectrum with the
                         receiver dwell window (◉ INTERCEPT / ✕ MISS), synthesized per-band
                         oscilloscope from the truth grid, waterfall, and a running hit-rate
                         duel vs the fixed sweep on the same scenario seed.
  3. Decision Inspector  live decision math: per-band UCB/Thompson score breakdown
                         (Q, exploration, due, priority, staleness), plain-English rationale,
                         Block-3 agent evidence, 6×6 band grid (click a band to lock the scope).
  4. Benchmark           paired-seed comparison of all 5 schedulers + full figures-of-merit table
                         (reads results/metrics_<split>.json when present, or runs fresh).

API (also usable standalone): POST /api/episode, GET /api/runs/{id}/truth.bin,
GET /api/runs/{id}/decision?step=N, GET /api/receiver, GET /api/benchmark.

Decision tracing lives in `server/trace.py` (score math copied from `b6_scheduler.select` —
keep in sync if that changes); `runner.run_episode` gained an optional `on_step` callback
(default None — existing callers unaffected).
* DQN/PPO are not implemented (bandit is the finalized core).v