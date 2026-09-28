// Global app store: episode data, truth grid, shared replay clock, decision fetch.
// One rAF-driven cursor (mission slot time) syncs every panel across all tabs.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { fetchDecision, fetchTruthBin, postEpisode } from './api'

const Ctx = createContext(null)

export function AppProvider({ children }) {
  // ---- scenario controls
  const [params, setParams] = useState({
    split: 'test', seedIndex: 0, algo: 'ucb', nEmitters: [4, 9],
  })
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)

  // ---- run payload (from POST /api/episode)
  const [run, setRun] = useState(null)          // {run_id, metrics, fixed_metrics, series, ...}
  const [truth, setTruth] = useState(null)      // Float32Array (n_bands*T), demo overlay
  const [bands, setBands] = useState(null)      // receiver info (centres, dwell times)

  // ---- playback (cursor = mission slot time, float)
  const [cursor, setCursor] = useState(0)       // slots 0..T
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(40)        // mission-seconds per wall-second
  const [selectedBand, setSelectedBand] = useState(null) // null = follow the scheduler
  const raf = useRef(0)
  const last = useRef(0)

  const T = run?.T ?? 4000
  const slotS = run?.slot_s ?? 0.05
  const series = run?.series
  const nDwells = series?.t?.length ?? 0

  // current dwell = last dwell whose start slot <= cursor
  const dwellIdx = useMemo(() => {
    if (!series?.t?.length) return -1
    const t = series.t
    let lo = 0, hi = t.length - 1, ans = 0
    if (cursor >= t[hi]) return hi
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (t[mid] <= cursor) { ans = mid; lo = mid + 1 } else hi = mid - 1
    }
    return ans
  }, [series, cursor])

  const dwell = useMemo(() => {
    if (dwellIdx < 0 || !series) return null
    return {
      idx: dwellIdx, t: series.t[dwellIdx], band: series.band[dwellIdx],
      hit: !!series.hit[dwellIdx], detected: !!series.detected[dwellIdx],
      dwellSlots: series.dwell[dwellIdx], latency: series.latency_ms[dwellIdx],
    }
  }, [series, dwellIdx])

  // ---- replay clock
  useEffect(() => {
    if (!playing || !run) return
    last.current = performance.now()
    const tick = (now) => {
      const dt = (now - last.current) / 1000
      last.current = now
      setCursor((c) => {
        const nc = c + dt * (speed / slotS)
        if (nc >= T) { setPlaying(false); return T }
        return nc
      })
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, run, speed, slotS, T])

  const play = useCallback(() => {
    setCursor((c) => (c >= T - 1 ? 0 : c))
    setPlaying(true)
  }, [T])
  const pause = useCallback(() => setPlaying(false), [])
  const replay = useCallback(() => { setCursor(0); setPlaying(true) }, [])
  const seekSlot = useCallback((s) => setCursor(Math.max(0, Math.min(T - 1, s))), [T])
  const seekDwell = useCallback((i) => {
    if (series?.t?.[i] != null) setCursor(series.t[i])
  }, [series])

  // ---- run an episode
  const runEpisode = useCallback(async (override = {}) => {
    const p = { ...params, ...override }
    setRunning(true)
    setError(null)
    setPlaying(false)
    try {
      const data = await postEpisode({
        split: p.split, seedIndex: p.seedIndex, algo: p.algo, nEmitters: p.nEmitters,
      })
      const truthGrid = await fetchTruthBin(data.run_id, data.n_bands, data.T)
      setRun(data)
      setTruth(truthGrid)
      setCursor(0)
      setPlaying(true)
      return data
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setRunning(false)
    }
  }, [params])

  // auto-run the first episode so the demo is never empty
  useEffect(() => { runEpisode() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const [decision, setDecision] = useState(null)
  const decTimer = useRef(0)
  // decision fetch: instant when scrubbing paused, throttled while playing
  useEffect(() => {
    if (!run || dwellIdx < 0) return
    clearTimeout(decTimer.current)
    const delay = playing ? 250 : 0
    decTimer.current = setTimeout(() => {
      fetchDecision(run.run_id, dwellIdx).then(setDecision).catch(() => {})
    }, delay)
    return () => clearTimeout(decTimer.current)
  }, [run, dwellIdx, playing])

  const value = {
    params, setParams, running, error,
    run, truth, bands, setBands,
    T, slotS, series, nDwells,
    cursor, playing, speed,
    play, pause, replay, seekSlot, seekDwell, setSpeed,
    dwellIdx, dwell,
    selectedBand, setSelectedBand,
    decision, runEpisode,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp() {
  return useContext(Ctx)
}

// cumulative hit-rate arrays (per dwell) for the running duel, memo-friendly
export function hitRateCurve(series) {
  if (!series) return null
  const out = new Float32Array(series.hit.length)
  let hits = 0
  for (let i = 0; i < series.hit.length; i++) {
    hits += series.hit[i] ? 1 : 0
    out[i] = hits / (i + 1)
  }
  return out
}
