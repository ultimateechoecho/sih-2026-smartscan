import AgentGrid from '../components/AgentGrid'
import DecisionBars from '../components/DecisionBars'
import LatencySparkline from '../components/LatencySparkline'
import Panel from '../components/Panel'
import PlaybackBar from '../components/PlaybackBar'
import RationaleCard from '../components/RationaleCard'
import { useApp } from '../store'

// Tab 3 — why did the scheduler pick that band? Live decision math, agent
// evidence and the plain-English rationale, synced to the replay.
export default function DecisionInspector() {
  const { decision, run } = useApp()

  return (
    <div className="space-y-3">
      <PlaybackBar />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel
          title="Decision rationale — current dwell"
          className="xl:col-span-2"
          right={
            decision && (
              <span className="text-[9px] text-[var(--dim)]">
                dwell #{decision.step + 1} · {decision.traced ? 'bandit trace' : 'baseline'}
              </span>
            )
          }
        >
          <RationaleCard />
        </Panel>

        <Panel title="Band state grid (36 dwell centres)">
          <AgentGrid />
        </Panel>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel
          title="Per-band score breakdown — top 8"
          className="xl:col-span-2"
          right={
            <span className="text-[9px] text-[var(--dim)]">
              score = Q + c·(1+w_hop·hop)·√(lnN/n) + w_due·due + w_pri·priority + w_stale·age
            </span>
          }
        >
          <DecisionBars />
        </Panel>

        <div className="flex flex-col gap-3">
          <Panel title="Block-3 agent evidence (chosen band)">
            {decision?.traced ? (
              <div className="space-y-1.5 text-[11px]">
                {['periodic_conf', 'scan_conf', 'stagger_conf'].map((k) => {
                  const v = decision.features[k][decision.action]
                  const label = { periodic_conf: 'Periodicity', scan_conf: 'Beam-scan', stagger_conf: 'Stagger' }[k]
                  return (
                    <div key={k}>
                      <div className="flex justify-between">
                        <span className="text-[var(--dim)]">{label}</span>
                        <span className="tabular-nums">{v.toFixed(2)}</span>
                      </div>
                      <div className="h-1.5 rounded bg-black/50 overflow-hidden">
                        <div className="h-full" style={{ width: `${v * 100}%`, background: '#fbbf24' }} />
                      </div>
                    </div>
                  )
                })}
                <div className="flex justify-between pt-1">
                  <span className="text-[var(--dim)]">Hopper suspicion (global)</span>
                  <span className="tabular-nums text-[var(--cyan)]">{(decision.hop_conf * 100).toFixed(0)}%</span>
                </div>
                <div className="text-[9px] text-[var(--dim)] pt-1 leading-relaxed">
                  Agents run pure physics (autocorrelation, FFT, envelope fit, interval pattern) on
                  observed hit history only — they advise, the bandit decides.
                </div>
              </div>
            ) : (
              <div className="text-[11px] text-[var(--dim)]">
                Baselines don't use agent evidence. Switch the scheduler to UCB or Thompson.
              </div>
            )}
          </Panel>
          <Panel title="Decision latency (ms / dwell)">
            <LatencySparkline height={64} />
          </Panel>
        </div>
      </div>
    </div>
  )
}
