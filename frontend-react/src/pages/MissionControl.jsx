import EmitterTable from '../components/EmitterTable'
import FoMGrid from '../components/FoMGrid'
import HeatmapCanvas from '../components/HeatmapCanvas'
import LatencySparkline from '../components/LatencySparkline'
import Panel from '../components/Panel'
import PlaybackBar from '../components/PlaybackBar'
import { useApp } from '../store'
import { fmtFom } from '../api'

// Tab 1 — mission replay: figures of merit, band×time heatmap with the scan
// trajectory, per-emitter intercepts.
export default function MissionControl() {
  const { run, error, dwell } = useApp()
  const m = run?.metrics

  return (
    <div className="space-y-3">
      <PlaybackBar />
      {error && (
        <div className="panel px-3 py-2 text-[var(--red)] text-[12px]">API error: {error}</div>
      )}

      <Panel title="Figures of merit — PS evaluation set (deltas vs fixed sweep, same seed)">
        {m ? <FoMGrid metrics={m} fixedMetrics={run.fixed_metrics} />
          : <div className="text-[var(--dim)] text-[12px] py-4">simulating mission…</div>}
      </Panel>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel
          title="Band occupancy × scan trajectory"
          className="xl:col-span-2"
          bodyClass="h-[380px]"
          right={
            <span className="text-[9px] text-[var(--dim)]">
              blue = emitter on air (hidden truth, demo) · green dot = HIT · grey = dwell · boxed = current
            </span>
          }
        >
          <HeatmapCanvas />
        </Panel>

        <div className="flex flex-col gap-3 min-h-0">
          <Panel title="Per-emitter intercepts" bodyClass="overflow-auto max-h-[230px]">
            <EmitterTable />
          </Panel>
          <Panel title="Decision latency (ms / dwell)">
            <LatencySparkline height={70} />
            {m && (
              <div className="text-[10px] text-[var(--dim)] mt-1">
                mean {fmtFom('latency_ms', m.latency_ms)} · {fmtFom('n_dwells', m.n_dwells)} dwells ·
                mean dwell {((run.series.dwell.reduce((a, b) => a + b, 0) / run.series.dwell.length) * run.slot_s).toFixed(2)} s
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  )
}
