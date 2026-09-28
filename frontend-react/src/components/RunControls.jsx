import { ALGOS, ALGO_LABEL, BASELINES } from '../api'
import { useApp } from '../store'

// Shared scenario controls — visible on Mission Control, Waveform Lab and
// Decision Inspector since they all replay the same episode.
export default function RunControls() {
  const { params, setParams, runEpisode, running, run } = useApp()
  const set = (k) => (e) => setParams((p) => ({ ...p, [k]: e.target.value }))
  const setEm = (i) => (e) =>
    setParams((p) => {
      const v = [...p.nEmitters]
      v[i] = +e.target.value
      return { ...p, nEmitters: v }
    })

  return (
    <div className="panel px-3 py-2 flex items-center gap-3 flex-wrap">
      <button className="btn-run" onClick={() => runEpisode()} disabled={running}>
        {running ? 'SIMULATING…' : '▶ RUN SCENARIO'}
      </button>

      <label className="flex items-center gap-1.5 text-[10px] tracking-widest text-[var(--dim)]">
        SPLIT
        <select className="field" value={params.split} onChange={set('split')}>
          {['test', 'val', 'train'].map((s) => <option key={s}>{s}</option>)}
        </select>
      </label>

      <label className="flex items-center gap-1.5 text-[10px] tracking-widest text-[var(--dim)]">
        EPISODE #
        <input type="number" min={0} max={9999} className="field w-16"
          value={params.seedIndex} onChange={(e) => setParams((p) => ({ ...p, seedIndex: +e.target.value }))} />
      </label>

      <label className="flex items-center gap-1.5 text-[10px] tracking-widest text-[var(--dim)]">
        EMITTERS
        <input type="number" min={2} max={14} className="field w-14" value={params.nEmitters[0]} onChange={setEm(0)} />
        <span className="text-[var(--dim)]">–</span>
        <input type="number" min={2} max={14} className="field w-14" value={params.nEmitters[1]} onChange={setEm(1)} />
      </label>

      <label className="flex items-center gap-1.5 text-[10px] tracking-widest text-[var(--dim)]">
        SCHEDULER
        <select className="field" value={params.algo} onChange={set('algo')}>
          {ALGOS.map((a) => <option key={a} value={a}>{ALGO_LABEL[a]}</option>)}
        </select>
      </label>

      {run && (
        <span className="ml-auto text-[10px] text-[var(--dim)]">
          {BASELINES.has(run.algo) ? 'baseline' : 'proposed'} · seed {run.seed} ·{' '}
          <span className={BASELINES.has(run.algo) ? '' : 'text-[var(--green-hi)]'}>{run.algo}</span>
        </span>
      )}
    </div>
  )
}
