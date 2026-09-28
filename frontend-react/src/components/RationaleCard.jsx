import { useApp } from '../store'
import { ALGO_LABEL } from '../api'

// Plain-English "why" for the current decision — the judge-facing explainability card.
export default function RationaleCard() {
  const { decision, run, slotS } = useApp()
  if (!decision || !run) return null
  const band = decision.action

  let headline
  if (decision.traced) {
    const c = decision.components
    const part = (k) => c[k][band]
    headline = (
      <>
        <span className="text-[var(--green-hi)] font-bold glow-green">Band {band}</span> selected —{' '}
        {run.algo === 'ucb' ? 'UCB' : 'Thompson'} score{' '}
        <span className="text-[var(--green-hi)] tabular-nums">{part('score').toFixed(2)}</span> ={' '}
        <span className="text-[var(--text)]">Q {part('Q').toFixed(2)}</span> +{' '}
        <span className="text-[var(--cyan)]">explore {part('explore').toFixed(2)}</span> +{' '}
        <span className="text-[var(--amber)]">due {part('due').toFixed(2)}</span> +{' '}
        <span className="text-[#f472b6]">priority {part('priority').toFixed(2)}</span> +{' '}
        <span className="text-[#94a3b8]">stale {part('stale').toFixed(2)}</span>
      </>
    )
  } else {
    headline = <span className="text-[var(--text)]">Band {band} — {decision.reason}</span>
  }

  // strongest agent evidence for the chosen band
  const f = decision.features
  const agents = [
    ['periodicity', f.periodic_conf[band]],
    ['beam-scan', f.scan_conf[band]],
    ['stagger', f.stagger_conf[band]],
  ].sort((a, b) => b[1] - a[1])
  const top = agents[0]
  const due = f.due[band]

  return (
    <div className="space-y-1.5 text-[12px] leading-relaxed">
      <div>{headline}</div>
      <div className="text-[11px] text-[var(--dim)]">
        at t = {(decision.t * slotS).toFixed(2)}s · decision took{' '}
        <span className="text-[var(--text)]">{decision.latency_ms.toFixed(3)} ms</span> · outcome{' '}
        <span className={decision.hit ? 'text-[var(--green-hi)] font-bold' : 'text-[var(--red)] font-bold'}>
          {decision.hit ? 'HIT' : 'MISS'}
        </span>
      </div>
      {decision.traced && (
        <div className="text-[11px]">
          Agent evidence:{' '}
          {top[1] > 0.05 ? (
            <>
              <span className="text-[var(--amber)]">{top[0]}</span> pattern detected (confidence{' '}
              {top[1].toFixed(2)})
              {due > 0.05 && <> · band flagged <b>due</b> now ({due.toFixed(2)})</>}
            </>
          ) : (
            'no strong pattern yet — exploring by confidence bound'
          )}
          {decision.hop_conf > 0.05 && (
            <> · hopper suspicion {(decision.hop_conf * 100).toFixed(0)}% widens exploration</>
          )}
        </div>
      )}
    </div>
  )
}
