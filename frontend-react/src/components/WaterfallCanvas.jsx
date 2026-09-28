import { useApp } from '../store'
import { useCanvas, dbNorm, GREEN, RED, DIM } from './canvasUtil'

// Waterfall (spectrogram-style): frequency bands stacked vertically, time
// scrolling right-to-left, colour = signal amplitude.  The receiver's dwell
// trajectory is traced on top.
export default function WaterfallCanvas() {
  const { run, truth, cursor, series } = useApp()
  const nBands = run?.n_bands ?? 36
  const T = run?.T ?? 4000

  const ref = useCanvas((ctx, w, h) => {
    const win = 600 // slots visible
    const t1 = Math.max(win, Math.min(T, Math.ceil(cursor) + 5))
    const t0 = t1 - win
    const rowH = h / nBands
    const img = ctx.createImageData(w, h)
    const d = img.data

    if (truth) {
      for (let x = 0; x < w; x++) {
        const t = t0 + Math.floor((x / w) * win)
        if (t < 0 || t >= T) continue
        for (let b = 0; b < nBands; b++) {
          const a = dbNorm(truth[b * T + t])
          if (a <= 0) continue
          const y0 = Math.floor(b * rowH), y1 = Math.floor((b + 1) * rowH)
          // cyan -> green -> yellow ramp
          const r = Math.floor(255 * Math.min(1, Math.max(0, (a - 0.55) * 2.2)))
          const g = Math.floor(120 + 135 * Math.min(1, a * 1.4))
          const bl = Math.floor(220 * (1 - a) + 30)
          for (let y = y0; y < Math.max(y0 + 1, y1); y++) {
            const i = (y * w + x) * 4
            d[i] = Math.max(d[i], r)
            d[i + 1] = Math.max(d[i + 1], g)
            d[i + 2] = Math.max(d[i + 2], bl)
            d[i + 3] = Math.floor(70 + 185 * a)
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0)

    // dwell trajectory overlay
    if (series) {
      const x = (t) => ((t - t0) / win) * w
      const y = (b) => (b + 0.5) * rowH
      for (let i = 0; i < series.t.length; i++) {
        const t = series.t[i]
        if (t < t0 || t > t1) continue
        ctx.fillStyle = series.hit[i] ? GREEN : 'rgba(200,220,212,0.35)'
        ctx.fillRect(x(t) - 0.5, y(series.band[i]) - rowH / 2, 1.5, rowH)
      }
      // now line
      ctx.strokeStyle = RED
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x(Math.min(cursor, t1)), 0)
      ctx.lineTo(x(Math.min(cursor, t1)), h)
      ctx.stroke()
    }

    ctx.fillStyle = DIM
    ctx.font = '9px ui-monospace, monospace'
    ctx.fillText(`${(t0 * 0.05).toFixed(0)}s`, 4, 10)
    ctx.fillText(`${(t1 * 0.05).toFixed(0)}s`, w - 30, 10)
  }, [run, truth, cursor, series])

  return <canvas ref={ref} className="w-full h-full block" />
}
