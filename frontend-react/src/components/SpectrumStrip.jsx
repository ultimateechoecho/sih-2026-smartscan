import { useApp } from '../store'
import { useCanvas, dbNorm, GREEN, GREEN_HI, RED, CYAN, DIM } from './canvasUtil'

// THE bandwidth hit/miss visualizer: all 36 bands across the frequency axis,
// live signal energy from the (demo) truth grid, and the receiver's dwell
// window sliding to whichever band the scheduler picked — HIT pulses green,
// MISS flashes red.
export default function SpectrumStrip() {
  const { run, truth, cursor, dwell, bands, selectedBand } = useApp()
  const nBands = run?.n_bands ?? 36
  const T = run?.T ?? 4000
  const slot = Math.min(T - 1, Math.floor(cursor))

  const ref = useCanvas((ctx, w, h) => {
    const padL = 8, padR = 8, padT = 16, padB = 20
    const cw = w - padL - padR, ch = h - padT - padB
    const bw = cw / nBands
    ctx.font = '8.5px ui-monospace, monospace'

    // baseline axis
    ctx.strokeStyle = 'rgba(52,211,153,0.25)'
    ctx.beginPath()
    ctx.moveTo(padL, padT + ch + 0.5)
    ctx.lineTo(w - padR, padT + ch + 0.5)
    ctx.stroke()

    // frequency tick labels (GHz) every 5th band
    ctx.fillStyle = DIM
    for (let b = 0; b < nBands; b += 5) {
      const ghz = ((bands?.centres_mhz?.[b] ?? b * 500 + 250) / 1000).toFixed(1)
      ctx.fillText(ghz, padL + b * bw + 1, h - 6)
    }
    ctx.fillText('GHz', w - padR - 22, h - 6)

    for (let b = 0; b < nBands; b++) {
      const x = padL + b * bw
      // dwell window for the band currently being scanned
      const isDwell = dwell && b === dwell.band
      const isSel = selectedBand != null && b === selectedBand
      if (isDwell || isSel) {
        ctx.fillStyle = isDwell ? 'rgba(52,211,153,0.10)' : 'rgba(56,189,248,0.08)'
        ctx.fillRect(x, padT, bw, ch)
        ctx.strokeStyle = isDwell ? GREEN : CYAN
        ctx.lineWidth = 1
        ctx.strokeRect(x + 0.5, padT + 0.5, bw - 1, ch - 1)
      }

      // live energy + peak-hold (trailing 40 slots ≈ 2 s persistence, like a
      // real spectrum analyzer's max-hold trace)
      let amp = -200
      if (truth) {
        const t0 = Math.max(0, slot - 40)
        for (let k = t0; k <= slot; k++) amp = Math.max(amp, truth[b * T + k])
      }
      const e = dbNorm(amp)
      if (e > 0) {
        const bh = e * (ch - 6)
        const grad = ctx.createLinearGradient(0, padT + ch - bh, 0, padT + ch)
        grad.addColorStop(0, 'rgba(110,231,183,0.95)')
        grad.addColorStop(1, 'rgba(52,211,153,0.25)')
        ctx.fillStyle = grad
        ctx.fillRect(x + 1.5, padT + ch - bh, bw - 3, bh)
      }

      // band centre marker
      ctx.fillStyle = 'rgba(95,122,112,0.6)'
      ctx.fillRect(x + bw / 2 - 0.5, padT + ch - 2, 1, 2)
    }

    // HIT / MISS verdict banner over the dwell window
    if (dwell) {
      const x = padL + dwell.band * bw
      const flash = dwell.hit ? GREEN : RED
      ctx.strokeStyle = flash
      ctx.lineWidth = 1.5
      const phase = (Date.now() % 800) / 800
      ctx.globalAlpha = 0.55 + 0.45 * Math.sin(phase * Math.PI * 2)
      ctx.strokeRect(x - 1.5, padT - 1.5, bw + 3, ch + 3)
      ctx.globalAlpha = 1
      ctx.fillStyle = flash
      ctx.font = 'bold 10px ui-monospace, monospace'
      const label = dwell.hit ? '◉ INTERCEPT' : '✕ MISS'
      const lx = Math.min(Math.max(x - 18, padL), w - padR - 78)
      ctx.fillText(label, lx, padT - 4)
    }
  }, [run, truth, cursor, dwell, bands, selectedBand])

  return <canvas ref={ref} className="w-full h-full block" />
}
