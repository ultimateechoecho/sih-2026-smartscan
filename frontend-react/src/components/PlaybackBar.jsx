import { useApp } from '../store'

const SPEEDS = [10, 40, 100, 200]

export default function PlaybackBar() {
  const {
    run, T, slotS, cursor, playing, speed, play, pause, replay, seekSlot, setSpeed, nDwells,
    dwellIdx, dwell,
  } = useApp()
  const missionS = cursor * slotS
  const totalS = T * slotS

  return (
    <div className="panel px-3 py-2 flex items-center gap-3 flex-wrap">
      <div className="flex items-center gap-1.5">
        <button className="btn-ghost" onClick={playing ? pause : play} disabled={!run}>
          {playing ? '❚❚ PAUSE' : '▶ PLAY'}
        </button>
        <button className="btn-ghost" onClick={replay} disabled={!run}>↻ REPLAY</button>
      </div>

      <div className="flex-1 min-w-[240px] flex items-center gap-2">
        <input
          type="range" min={0} max={T - 1} value={cursor}
          onChange={(e) => seekSlot(+e.target.value)}
          className="w-full" disabled={!run}
        />
        <span className="text-[11px] text-[var(--dim)] whitespace-nowrap tabular-nums">
          t = {missionS.toFixed(1)}s / {totalS.toFixed(0)}s
        </span>
      </div>

      <div className="flex items-center gap-1">
        <span className="text-[10px] tracking-widest text-[var(--dim)]">SPEED</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            className={`px-2 py-0.5 rounded text-[10px] border ${
              speed === s
                ? 'border-[var(--green)] text-[var(--green-hi)] bg-[rgba(52,211,153,0.12)]'
                : 'border-transparent text-[var(--dim)] hover:text-[var(--green-hi)]'
            }`}
          >
            {s}×
          </button>
        ))}
      </div>

      {dwell && (
        <div className="flex items-center gap-2 text-[11px] tabular-nums">
          <span className="text-[var(--dim)]">dwell</span>
          <span className="text-[var(--text)]">{dwellIdx + 1}/{nDwells}</span>
          <span className="text-[var(--dim)]">band</span>
          <span className="text-[var(--cyan)]">{dwell.band}</span>
          <span
            className={`px-1.5 rounded font-bold ${
              dwell.hit
                ? 'text-[#052e21] bg-[var(--green)]'
                : 'text-[var(--red)] bg-[rgba(251,113,133,0.12)]'
            }`}
          >
            {dwell.hit ? 'HIT' : 'MISS'}
          </span>
        </div>
      )}
    </div>
  )
}
