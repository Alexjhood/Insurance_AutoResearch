import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { MetricNumber } from '../../components/MetricNumber';
import { PageState } from '../../components/PageState';
import { StatTile } from '../../components/StatTile';
import { TokenFlow } from '../../exhibits/TokenFlow';
import { compact, duration, totalTokens } from '../../lib/format';
import { durationLabel } from '../../exhibits/shared';
import { useSnapshot } from '../../lib/data/queries';
import './Telemetry.css';

export function Telemetry() {
  const { orchId } = useParams();
  const q = useSnapshot(orchId);
  const s = q.data;
  const [group, setGroup] = useState<'delegation' | 'model'>('delegation');
  if (!s) return <PageState loading={q.isLoading} error={q.error} />;
  const maxCalls = Math.max(...s.telemetry_summary.tool_mix.map(x => x.calls), 1);
  const promoted = s.experiments.filter(e => e.comparison?.decision === 'promote');
  const lift = promoted.reduce((n, e) => n + Math.max(e.lift.vs_then_champion ?? 0, 0), 0);
  const fit = s.experiments.reduce((n, e) => n + (e.metrics.fit_wall_seconds ?? 0), 0);
  const modelRows = s.telemetry_summary.usage_by_model.filter(x => x.tokens).map(x => ({
    delegation_id: `${x.model}${x.effort ? ` · ${x.effort}` : ''}`,
    tokens: x.tokens!,
    model_calls: 0, tool_calls: 0, tool_failures: 0,
    cache_hit_rate: x.tokens!.input ? x.tokens!.cached_input / x.tokens!.input : null,
  }));
  const unmeasured = s.telemetry_summary.usage_by_model.filter(x => x.unmeasured);
  return <main className="page telemetry-page">
    <header><span className="eyebrow">Resource analytics</span><h1>Telemetry</h1><p>Token, tool, turn, and model-efficiency evidence across the campaign.</p></header>
    <div className="telemetry-grouping"><button className={group==='delegation'?'active':''} onClick={()=>setGroup('delegation')}>By delegation</button><button className={group==='model'?'active':''} onClick={()=>setGroup('model')}>By model</button></div>
    <TokenFlow mode="campaign" byDelegation={group === 'delegation' ? s.telemetry_summary.by_delegation : modelRows} />
    {group === 'model' && unmeasured.map(x => <p className="unmeasured-model panel" key={x.key}>{x.role === 'orchestrator' ? 'Orchestrator' : x.role}: {x.model}{x.effort ? ` · ${x.effort}` : ''} · usage unmeasured</p>)}
    <section className="panel tool-mix"><header className="section-title"><div><span className="eyebrow">Tool-call mix</span><h2>Calls and occupied time</h2></div></header>{s.telemetry_summary.tool_mix.map(x => <Link className="tool-bar-row" key={`${x.delegation_id}-${x.tool}`} to={`/o/${orchId}/delegations/${x.delegation_id}`}><b>{x.delegation_id}</b><span>{x.tool}</span><div><i style={{ width: `${x.calls / maxCalls * 100}%` }} /><i className="failure" style={{ width: `${x.failures / Math.max(x.calls, 1) * 100}%` }} /></div><span className="mono">{x.calls} · {durationLabel(x.total_duration_ms)}</span></Link>)}</section>
    <div className="telemetry-grid"><section className="panel economics"><header className="section-title"><div><span className="eyebrow">Turn economics</span><h2>Cache and model-call trend</h2></div></header><svg viewBox="0 0 560 220" role="img" aria-label="Cache hit trend and model calls per cycle">{s.telemetry_summary.by_delegation.map((d, i) => { const cycles=s.delegations.find(x=>x.delegation_id===d.delegation_id)?.budget.decided??1; return <g key={d.delegation_id}><rect x={42+i*100} y={190-(d.cache_hit_rate??0)*150} width="28" height={(d.cache_hit_rate??0)*150}/><circle cx={56+i*100} cy={190-(d.model_calls/Math.max(cycles,1))*3} r="5"/><text x={45+i*100} y="210">{d.delegation_id}</text></g>; })}<line x1="30" y1="190" x2="540" y2="190"/></svg><p>Bars: cache-hit rate · points: model calls per decided cycle</p></section>
      <section className="panel efficiency"><header className="section-title"><div><span className="eyebrow">Efficiency panel</span><h2>Campaign yield</h2></div></header><div className="efficiency-grid"><StatTile label="Estimated campaign cost" value={s.telemetry_summary.cost_usd == null ? '—' : `$${s.telemetry_summary.cost_usd.toFixed(2)} estimated`} /><StatTile label="Tokens / promoted lift point" value={lift ? compact(totalTokens(s.telemetry_summary.totals) / lift) : '—'} /><StatTile label="Wall clock / decided cycle" value={duration(s.delegations.reduce((n,d)=>n+(d.cost.wall_clock_minutes??0),0)/Math.max(s.delegations.reduce((n,d)=>n+d.budget.decided,0),1))} /><StatTile label="Screening + CV fit time" value={duration(fit/60)} /><StatTile label="Tool failures" value={s.telemetry_summary.totals.tool_failures} /><StatTile label="Cache hit" value={<MetricNumber kind="percent" value={s.telemetry_summary.cache_hit_rate} />} /></div></section>
    </div>
  </main>;
}
