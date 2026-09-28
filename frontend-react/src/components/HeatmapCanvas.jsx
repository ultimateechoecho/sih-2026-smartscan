import { useApp } from '../store'
import { useCanvas, GREEN, RED, CYAN, DIM } from './canvasUtil'

const BIN = 10 // slots per heatmap column (~0.5 s)

// Band × time occupancy (hidden truth, demo overlay) + the receiver's dwell
// trajectory up to the current replay position.
export default function HeatmapCanvas() {
  const { run, truth, cursor, series, dwellIdx, dwell } = useApp()
  const nBands = run?.n_bands ?? 36
  const T = run?.T ?? 4000

  const ref = useCanvas((ctx, w, h) => {
    const padL = 30, padR = 8, padT = 6, padB = 18
    const cw = w - padL - padR, ch = h - padT - padB
    const rowH = ch / nBands
    ctx.font = '9px ui-monospace, monospace'

    // truth occupancy (binned max)
    if (truth) {
      const nCols = Math.floor(T / BIN)
      for (let b = 0; b < nBands; b++) {
        for (let c = 0; c < nCols; c++) {
          let m = -200
          const off = b * T + c * BIN
          for (let k = 0; k < BIN; k++) m = Math.max(m, truth[off + k])
          if (m > -199) {
            ctx.fillStyle = `rgba(56,189,248,${0.10 + 0.22 * Math.min(1, (m + 70) / 40)})`
            ctx.fillRect(padL + (c * cw) / nCols, padT + b * rowH, cw / nCols + 0.5, rowH + 0.5)
          }
        }
      }
    }

    // dwell trajectory up to cursor
    if (series) {
      const x = (t) => padL + ((t + 0.5) / T) * cw
      const y = (b) => padT + (b + 0.5) * rowH
      const upto = dwellIdx + 1
      ctx.strokeStyle = 'rgba(95,122,112,0.22)'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let i = 0; i < upto; i++) {
        const px = x(series.t[i]), py = y(series.band[i])
        if (i === 0) ctx.moveTo(px, py)
        else {
          const prevX = x(series.t[i - 1] + series.dwell[i - 1])
          ctx.lineTo(prevX, py)
          ctx.lineTo(px, py)
        }
      }
      ctx.stroke()

      for (let i = 0; i < upto; i++) {
        const hit = series.hit[i]
        ctx.fillStyle = hit ? GREEN : 'rgba(148,180,168,0.45)'
        ctx.beginPath()
        ctx.arc(x(series.t[i]), y(series.band[i]), hit ? 2.6 : 1.3, 0, Math.PI * 2)
        ctx.fill()
      }
      // current dwell marker
      if (dwell) {
        const px = x(dwell.t), py = y(dwell.band)
        ctx.strokeStyle = dwell.hit ? GREEN : RED
        ctx.lineWidth = 1.4
        ctx.strokeRect(px - 4, py - 4, 8, 8)
        ctx.beginPath()
        ctx.moveTo(px, padT)
        ctx.lineTo(px, padT + ch)
        ctx.setLineDash([3, 4])
        ctx.strokeStyle = dwell.hit ? 'rgba(52,211,153,0.5)' : 'rgba(251,113,133,0.5)'
        ctx.stroke()
        ctx.setLineDash([])
      }
    }

    // axes
    ctx.fillStyle = DIM
    for (let b = 0; b < nBands; b += 6) ctx.fillText(String(b).padStart(2, '0'), 6, padT + (b + 0.7) * rowH)
    const totalS = T * 0.05
    for (let s = 0; s <= totalS; s += 25) {
      const px = padL + ((s / 0.05) / T) * cw
      ctx.fillText(`${s}s`, Math.min(px, w - 22), h - 5)
    }
  }, [run, truth, cursor, series, dwellIdx, dwell])

  return <canvas ref={ref} className="w-full h-full block" />
}
