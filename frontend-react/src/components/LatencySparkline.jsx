import { useApp } from '../store'
import { useCanvas, GREEN, DIM } from './canvasUtil'

// Scheduler compute time per decision (ms) across the mission so far.
export default function LatencySparkline({ height = 60 }) {
  const { series, dwellIdx } = useApp()
  const ref = useCanvas((ctx, w, h) => {
    if (!series?.latency_ms?.length) return
    const n = series.latency_ms.length
    const max = Math.max(...series.latency_ms) * 1.15
    const px = (i) => (i / n) * w
    const py = (v) => h - 4 - (v / max) * (h - 10)
    ctx.strokeStyle = GREEN
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let i = 0; i <= Math.min(dwellIdx, n - 1); i++) {
      const y = py(series.latency_ms[i])
      i === 0 ? ctx.moveTo(px(i), y) : ctx.lineTo(px(i), y)
    }
    ctx.stroke()
    ctx.fillStyle = DIM
    ctx.font = '9px ui-monospace, monospace'
    ctx.fillText(`max ${max.toFixed(2)} ms`, 4, 10)
    const cur = series.latency_ms[Math.min(dwellIdx, n - 1)]
    ctx.fillStyle = GREEN
    ctx.beginPath()
    ctx.arc(px(Math.min(dwellIdx, n - 1)), py(cur), 2.2, 0, Math.PI * 2)
    ctx.fill()
  }, [series, dwellIdx, height])
  return <canvas ref={ref} className="w-full block" style={{ height }} />
}
