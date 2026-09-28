// API client + figures-of-merit metadata (shared by Mission Control and Benchmark).

export const ALGOS = ['fixed', 'random', 'greedy', 'ucb', 'thompson']
export const BASELINES = new Set(['fixed', 'random', 'greedy'])

export const ALGO_LABEL = {
  fixed: 'Fixed sweep',
  random: 'Random scan',
  greedy: 'Greedy revisit',
  ucb: 'UCB bandit',
  thompson: 'Thompson sampling',
}

// Figures of merit — the PS's own evaluation set. better: -1 = higher is better,
// 1 = lower is better. pct: render as percentage.
export const FOM = [
  { key: 'pd', label: 'Pd — Probability of Detection', short: 'Pd', unit: '%', pct: true, better: -1,
    def: 'Share of emitters intercepted at least once during the mission.' },
  { key: 'interception_ratio', label: 'Interception Ratio', short: 'Ratio', unit: '%', pct: true, better: -1,
    def: 'Emitter bursts intercepted / all emitter bursts on air.' },
  { key: 'sensitivity', label: 'Sensitivity', short: 'Sens', unit: '%', pct: true, better: -1,
    def: 'Dwell-level detection probability given an emitter was active (TP / (TP+FN)).' },
  { key: 'pfa', label: 'Pfa — False Alarm Rate', short: 'Pfa', unit: '%', pct: true, better: 1,
    def: 'False alarms / dwells on bands with no emitter on air (lower is better).' },
  { key: 'intercept_rate_10s', label: 'Intercept Rate', short: 'Rate', unit: '/10s', better: -1,
    def: 'Emitters intercepted per 10 seconds of mission time.' },
  { key: 'intercept_delay_cens_s', label: 'Intercept Delay', short: 'Delay', unit: 's', better: 1,
    def: 'Mean time from an emitter switching on to its first interception. Un-intercepted emitters count as full mission length (censored mean, lower is better).' },
  { key: 'avg_reward', label: 'Avg Episode Reward', short: 'Reward', unit: '', better: -1,
    def: 'Total evaluation reward over the 200 s mission (truth-aware scoring).' },
  { key: 'pct_correct_pred', label: 'Prediction Accuracy', short: 'Pred', unit: '%', pct: true, better: -1,
    def: 'Share of agent predictions (next activity time per band) confirmed by ground truth (±4 slots).' },
  { key: 'n_dwells', label: 'Dwells', short: 'Dwells', unit: '', better: null,
    def: 'Number of receiver dwells (decisions) in the mission.' },
  { key: 'latency_ms', label: 'Decision Latency', short: 'Latency', unit: 'ms', better: 1,
    def: 'Mean scheduler compute time per decision (lower is better).' },
]

export const PRIMARY_FOM = ['pd', 'interception_ratio', 'intercept_delay_cens_s', 'avg_reward']

export function fmtFom(key, v) {
  const m = FOM.find((f) => f.key === key)
  if (v == null || Number.isNaN(v)) return '—'
  if (m?.pct) return (v * 100).toFixed(1) + '%'
  if (key === 'latency_ms') return v.toFixed(3) + ' ms'
  if (key === 'n_dwells') return String(Math.round(v))
  if (key === 'avg_reward') return v.toFixed(0)
  if (key === 'intercept_rate_10s') return v.toFixed(2)
  return v.toFixed(2) + ' s'
}

export async function getHealth() {
  const r = await fetch('/api/health')
  if (!r.ok) throw new Error('API down')
  return r.json()
}

export async function getReceiver() {
  const r = await fetch('/api/receiver')
  if (!r.ok) throw new Error('receiver fetch failed')
  return r.json()
}

export async function postEpisode({ split, seedIndex, algo, nEmitters }) {
  const r = await fetch('/api/episode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ split, seed_index: seedIndex, algo, n_emitters: nEmitters }),
  })
  if (!r.ok) throw new Error((await r.json()).detail || 'episode failed')
  return r.json()
}

export async function fetchTruthBin(runId, nBands, T) {
  const r = await fetch(`/api/runs/${runId}/truth.bin`)
  if (!r.ok) throw new Error('truth fetch failed')
  const buf = await r.arrayBuffer()
  return new Float32Array(buf, 0, nBands * T) // row-major (n_bands, T), -200 = silent
}

export async function fetchDecision(runId, step) {
  const r = await fetch(`/api/runs/${runId}/decision?step=${step}`)
  if (!r.ok) throw new Error('decision fetch failed')
  return r.json()
}

export async function fetchBenchmark(split, episodes, useCache = true) {
  const r = await fetch(`/api/benchmark?split=${split}&episodes=${episodes}&use_cache=${useCache}`)
  if (!r.ok) throw new Error('benchmark fetch failed')
  return r.json()
}
