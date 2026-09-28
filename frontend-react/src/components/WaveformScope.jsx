import { useApp } from '../store'
import { useCanvas, dbNorm, GREEN, RED, DIM, CYAN } from './canvasUtil'

const WIN = 400 // slots shown (~20 s)

// Oscilloscope for the selected band: the waveform is synthesized from the
// truth grid — pulse bursts (on/off keying) whose amplitude follows the
// emitter's physics (scan emitters rise/fall smoothly, hoppers burst and
// vanish).  Dwell verdicts are marked along the bottom.
export default function WaveformScope() {
  const { run, truth, cursor, dwell, series, bands, selectedBand, setSelectedBand } = useApp()
  const nBands = run?.n_bands ?? 36
  const T = run?.T ?? 4000
  const band = selectedBand ?? dwell?.band ?? 0
  const t1 = Math.max(WIN, Math.min(T, Math.ceil(cursor) + 20))
  const t0 = t1 - WIN
  const centre = bands?.centres_mhz?.[band] ?? band * 500 + 250

  const ref = useCanvas((ctx, w, h) => {
    const padT = 14, padB = 22
    const ch = h - padT - padB
    const mid = padT + ch / 2
    ctx.font = '9px ui-monospace, monospace'

    // graticule
    ctx.strokeStyle = 'rgba(52,211,153,0.08)'
    ctx.lineWidth = 1
    for (let i = 1; i < 6; i++) {
      const y = padT + (ch * i) / 6
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke()
    }
    for (let i = 1; i < 10; i++) {
      const x = (w * i) / 10
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ch); ctx.stroke()
    }

    // waveform: on/off-keyed carrier, amplitude from truth dB
    if (truth) {
      const px = (t) => ((t - t0) / WIN) * w
      ctx.strokeStyle = GREEN
      ctx.lineWidth = 1
      ctx.shadowColor = 'rgba(52,211,153,0.7)'
      ctx.shadowBlur = 4
      ctx.beginPath()
      let prevAmp = 0
      for (let x = 0; x <= w; x++) {
        const t = t0 + (x / w) * WIN
        const a = dbNorm(truth[band * T + Math.min(T - 1, Math.floor(t))])
        prevAmp = prevAmp * 0.6 + a * 0.4 // smooth edges slightly
        const carrier = Math.sin((x / w) * WIN * 0.9 + centre) // visual carrier
        const noise = (Math.random() - 0.5) * 0.05
        const y = mid - (prevAmp * 0.92 + noise) * (ch / 2) * carrier * 0.9
        if (x === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
      ctx.shadowBlur = 0

      // envelope hint
      ctx.strokeStyle = 'rgba(56,189,248,0.35)'
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      for (let t = t0; t <= t1; t += 4) {
        const a = dbNorm(truth[band * T + Math.min(T - 1, t)])
        const y = mid - a * (ch / 2) * 0.92
        const x = px(t)
        if (t === t0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
      ctx.setLineDash([])
    }

    // dwell verdict markers
    if (series) {
      for (let i = 0; i < series.t.length; i++) {
        const t = series.t[i]
        if (t < t0 || t > t1) continue
        if (series.band[i] !== band) continue
        const x = ((t - t0) / WIN) * w
        const hit = series.hit[i]
        ctx.fillStyle = hit ? GREEN : RED
        ctx.beginPath()
        ctx.arc(x, padT + ch + 9, hit ? 3.2 : 2, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // now-line
    const nx = ((Math.min(cursor, t1) - t0) / WIN) * w
    ctx.strokeStyle = 'rgba(251,191,36,0.7)'
    ctx.beginPath(); ctx.moveTo(nx, padT); ctx.lineTo(nx, padT + ch); ctx.stroke()

    // labels
    ctx.fillStyle = DIM
    ctx.fillText(`band ${String(band).padStart(2, '0')} · ${(centre / 1000).toFixed(2)} GHz`, 6, 10)
    ctx.fillText(`${((t0 * 0.05)).toFixed(0)} s`, 6, h - 8)
    ctx.fillText(`${((t1 * 0.05)).toFixed(0)} s`, w - 34, h - 8)
    ctx.fillStyle = CYAN
    ctx.fillText('20 s window', w / 2 - 30, 10)
  }, [run, truth, cursor, dwell, series, band, bands, selectedBand, t0, t1])

  return (
    <div className="relative h-full">
      <canvas ref={ref} className="w-full h-full block" />
      <select
        aria-label="scope band"
        className="field absolute top-6 right-2 text-[10px] py-0.5"
        value={selectedBand ?? ''}
        onChange={(e) => setSelectedBand(e.target.value === '' ? null : +e.target.value)}
      >
        <option value="">follow scheduler</option>
        {Array.from({ length: nBands }, (_, b) => (
          <option key={b} value={b}>band {String(b).padStart(2, '0')}</option>
        ))}
      </select>
    </div>
  )
}
