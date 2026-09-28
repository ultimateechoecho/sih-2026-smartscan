import { useApp } from '../store'

// Per-emitter intercept table: who was on air, when we caught them, burst coverage.
export default function EmitterTable() {
  const { run } = useApp()
  const ems = run?.emitters ?? []
  return (
    <table className="roc w-full text-[11px] tabular-nums">
      <thead>
        <tr>
          <th>EID</th><th>Kind</th><th>On air</th><th>Intercepted</th>
          <th>Bursts</th><th>Bursts int.</th>
        </tr>
      </thead>
      <tbody>
        {ems.map((e) => (
          <tr key={e.eid}>
            <td className="text-[var(--cyan)]">EM-{String(e.eid).padStart(2, '0')}</td>
            <td className="text-[var(--amber)]">{e.kind}</td>
            <td>{e.first_active != null ? `${(e.first_active * 0.05).toFixed(1)}s` : '—'}</td>
            <td className={e.first_detect != null ? 'text-[var(--green-hi)]' : 'text-[var(--red)]'}>
              {e.first_detect != null ? `${(e.first_detect * 0.05).toFixed(1)}s` : 'NOT INTERCEPTED'}
            </td>
            <td>{e.n_bursts}</td>
            <td>{e.n_bursts_det}</td>
          </tr>
        ))}
        {!ems.length && <tr><td colSpan={6} className="text-[var(--dim)]">run an episode…</td></tr>}
      </tbody>
    </table>
  )
}
