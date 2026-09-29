"""FastAPI bridge — serves the smartscan engine to the React frontend.

Run from the project root:
    venv/Scripts/python -m uvicorn server.api:app --port 8000

After `npm run build` in frontend-react/, the built app is served from "/" so a
judge demo needs only this one command.  During development, run Vite's dev
server (npm run dev) and it proxies /api to this process.
"""
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from smartscan.b1_data import load_receiver
from smartscan.b6_scheduler import ALGOS, load_tuned
from smartscan.config import EnvConfig, RESULTS_DIR, SEED_BASE
from smartscan.evaluate import run_benchmark
from smartscan.metrics import ORDER, episode_metrics
from smartscan.runner import make_env, run_episode
from server.trace import make_traced_scheduler, run_episode_traced

app = FastAPI(title="SmartScan API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

RUNS = {}          # run_id -> episode payload + truth grid + trace
_ENVS = {}         # (split, n_lo, n_hi) -> SmartScanEnv (env reset() re-rolls emitters)


def _get_env(split, n_lo, n_hi):
    key = (split, n_lo, n_hi)
    if key not in _ENVS:
        _ENVS[key] = make_env(split, EnvConfig(n_emitters=(n_lo, max(n_lo + 1, n_hi))))
    return _ENVS[key]


class EpisodeReq(BaseModel):
    split: str = "test"
    seed_index: int = 0
    algo: str = "ucb"
    n_emitters: list[int] = Field(default=[4, 9], min_length=2, max_length=2)


@app.get("/api/health")
def health():
    return {"ok": True, "algos": list(ALGOS)}


@app.get("/api/receiver")
def receiver():
    rec = load_receiver()
    return {"n_bands": len(rec["dwell_centres_mhz"]),
            "centres_mhz": rec["dwell_centres_mhz"],
            "dwell_times_s": rec["dwell_times_s"],
            "source": rec.get("source", "")}


@app.post("/api/episode")
def episode(req: EpisodeReq):
    if req.algo not in ALGOS:
        raise HTTPException(400, f"algo must be one of {list(ALGOS)}")
    if req.split not in SEED_BASE:
        raise HTTPException(400, f"split must be one of {list(SEED_BASE)}")
    n_lo, n_hi = sorted(int(x) for x in req.n_emitters)
    env = _get_env(req.split, n_lo, n_hi)
    seed = SEED_BASE[req.split] + int(req.seed_index)

    params = load_tuned(req.algo) if req.algo in ("ucb", "thompson") else None
    log, trace = run_episode_traced(env, make_traced_scheduler(req.algo, params), seed)
    metrics = episode_metrics(log)

    # same seed, same scenario through the open-loop baseline -> the impact delta
    fixed_log, _ = run_episode_traced(env, make_traced_scheduler("fixed"), seed)
    fixed_metrics = episode_metrics(fixed_log)

    truth = env.truth_grid()
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = {"log": log, "trace": trace, "truth": truth,
                    "metrics": metrics, "fixed_metrics": fixed_metrics,
                    "req": req.model_dump(), "seed": seed}
    if len(RUNS) > 8:                                   # keep memory bounded
        RUNS.pop(next(iter(RUNS)))

    return {"run_id": run_id, "algo": req.algo, "split": req.split, "seed": seed,
            "T": log.T, "slot_s": log.slot_s, "n_bands": truth.shape[0],
            "metrics": metrics, "fixed_metrics": fixed_metrics,
            "emitters": log.emitters,
            "series": {"t": log.t, "band": log.band, "hit": log.hit,
                       "detected": log.detected, "active": log.active,
                       "fa": log.fa, "dwell": log.dwell,
                       "latency_ms": [round(x, 4) for x in log.latency_ms]},
            "fixed_series": {"t": fixed_log.t, "band": fixed_log.band,
                             "hit": fixed_log.hit, "dwell": fixed_log.dwell}}


@app.get("/api/runs/{run_id}/truth.bin")
def truth_bin(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id (episode runs live in memory; re-run it)")
    arr = np.ascontiguousarray(run["truth"], dtype=np.float32)
    return Response(content=arr.tobytes(), media_type="application/octet-stream")


@app.get("/api/runs/{run_id}/decision")
def decision(run_id: str, step: int = Query(ge=0)):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")
    log, trace = run["log"], run["trace"]
    if step >= len(log.t):
        raise HTTPException(400, f"step must be < {len(log.t)}")
    dec = trace["decisions"][step]
    F = trace["features"][step]
    names = ["hit_rate", "visit_age", "hit_age", "periodic_conf",
             "due", "scan_conf", "stagger_conf", "priority"]
    has_scores = "Q" in dec            # bandits carry score components; baselines only a reason
    return {"step": step, "t": log.t[step], "action": log.band[step],
            "hit": log.hit[step], "detected": log.detected[step],
            "dwell_slots": log.dwell[step], "latency_ms": log.latency_ms[step],
            "traced": has_scores,
            "reason": dec.get("reason"),
            "components": ({k: dec[k] for k in ("Q", "explore", "due", "priority",
                                                "stale", "score")}
                           if has_scores else None),
            "hop_conf": dec.get("hop_conf", 0.0),
            "features": {n: F[:, i].tolist() for i, n in enumerate(names)}}


@app.get("/api/benchmark")
def benchmark(split: str = "test", episodes: int = 5, use_cache: bool = True):
    if split not in SEED_BASE:
        raise HTTPException(400, f"split must be one of {list(SEED_BASE)}")
    cached = RESULTS_DIR / f"metrics_{split}.json"
    if use_cache and cached.exists():
        import json
        data = json.loads(cached.read_text())
        return {"cached": True, "split": data["split"], "episodes": data["episodes"],
                "algos": list(data["metrics"]),
                "mean": {a: {k: v[0] for k, v in m.items()}
                         for a, m in data["metrics"].items()},
                "std": {a: {k: v[1] for k, v in m.items()}
                        for a, m in data["metrics"].items()}}
    table, _ = run_benchmark(list(ALGOS), split, max(1, min(episodes, 20)), jobs=1)
    return {"cached": False, "split": split, "episodes": episodes,
            "algos": list(ALGOS),
            "mean": {a: {k: v[0] for k, v in m.items()} for a, m in table.items()},
            "std": {a: {k: v[1] for k, v in m.items()} for a, m in table.items()}}


# ---- serve the built React app (single-port demo) --------------------------
_DIST = ROOT / "frontend-react" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
