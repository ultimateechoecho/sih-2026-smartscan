import HitRateDuel from '../components/HitRateDuel'
import Panel from '../components/Panel'
import PlaybackBar from '../components/PlaybackBar'
import SpectrumStrip from '../components/SpectrumStrip'
import WaterfallCanvas from '../components/WaterfallCanvas'
import WaveformScope from '../components/WaveformScope'
import { useApp } from '../store'

// Tab 2 — the bandwidth hit/miss visualizer: live spectrum strip with the
// receiver's dwell window, oscilloscope of the scanned band, waterfall, and
// the running hit-rate duel vs the fixed-sweep baseline.
export default function WaveformLab() {
  const { run, selectedBand, setSelectedBand, dwell } = useApp()

  return (
    <div className="space-y-3">
      <PlaybackBar />

      <Panel
        title="RF spectrum — 36 receiver bands · bandwidth hit/miss visualizer"
        right={
          <span className="text-[9px] text-[var(--dim)]">
            bars = on-air energy (truth, demo) · green window = receiver dwell · ◉ INTERCEPT / ✕ MISS
          </span>
        }
        bodyClass="h-[190px]"
      >
        <SpectrumStrip />
      </Panel>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <Panel
          title="Oscilloscope — scanned band"
          className="xl:col-span-2"
          bodyClass="h-[240px]"
          right={
            <div className="flex items-center gap-2 text-[10px]">
              {selectedBand != null && (
                <button className="btn-ghost" onClick={() => setSelectedBand(null)}>unlock (follow scheduler)</button>
              )}
              <span className="text-[var(--dim)]">
                {selectedBand != null ? `locked B${selectedBand}` : dwell ? `following B${dwell.band}` : ''}
              </span>
            </div>
          }
        >
          <WaveformScope />
        </Panel>

        <div className="flex flex-col gap-3">
          <Panel title="Hit-rate duel — same scenario seed">
            <HitRateDuel />
          </Panel>
          <Panel title="Waterfall (time scrolling right)" bodyClass="h-[160px]">
            <WaterfallCanvas />
          </Panel>
        </div>
      </div>
    </div>
  )
}
