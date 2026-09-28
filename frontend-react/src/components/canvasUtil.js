import { useEffect, useRef } from 'react'

// Canvas hook: handles DPR sizing + ResizeObserver; `draw(ctx, w, h)` runs on
// every dependency change (call it inside requestAnimationFrame-sensitive paths).
export function useCanvas(draw, deps) {
  const ref = useRef(null)
  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    let raf = 0
    const render = () => {
      raf = 0
      const dpr = window.devicePixelRatio || 1
      const w = cv.clientWidth, h = cv.clientHeight
      if (!w || !h) return
      if (cv.width !== w * dpr || cv.height !== h * dpr) {
        cv.width = w * dpr
        cv.height = h * dpr
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      draw(ctx, w, h)
    }
    raf = requestAnimationFrame(render)
    const ro = new ResizeObserver(() => { if (!raf) raf = requestAnimationFrame(render) })
    ro.observe(cv)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return ref
}

// dB (-200..0) -> [0,1] visual energy, floored at the noise bottom
export function dbNorm(db, lo = -110, hi = -30) {
  if (db <= -199) return 0
  return Math.max(0, Math.min(1, (db - lo) / (hi - lo)))
}

export const GREEN = '#34d399'
export const GREEN_HI = '#6ee7b7'
export const RED = '#fb7185'
export const AMBER = '#fbbf24'
export const CYAN = '#38bdf8'
export const DIM = '#2a4a3f'
