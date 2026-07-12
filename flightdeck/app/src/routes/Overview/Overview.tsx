import { Link, useParams } from 'react-router-dom';
import { Badge } from '../../components/Badge';
import { MetricNumber } from '../../components/MetricNumber';
import { PageState } from '../../components/PageState';
import { StatTile } from '../../components/StatTile';
import { ChampionAscent } from '../../exhibits/ChampionAscent';
import { ResearchTree } from '../../exhibits/ResearchTree';
import { championDescriptor, compact, duration, orchestratorLabel, totalTokens } from '../../lib/format';
import { useSnapshot } from '../../lib/data/queries';
import './Overview.css';

const DISTRESS_COPY: Record<string, string> = {
  early_stop: 'Stopped early because the brief stop condition was satisfied; unused budget was preserved.',
  auto_rejected: 'The framework screening gate rejected one or more candidates automatically.',
  cycles_forfeited: 'Budgeted cycles were lost to interrupted or nonterminal work.',
  all_rejected: 'Every decided challenger in this delegation was rejected.',
  budget_overrun: 'The delegation exceeded its expected time or compute envelope.',
};

export function Overview() {
  const { orchId } = useParams();
  const q = useSnapshot(orchId);
  const s = q.data;
  if (!s) return <PageState loading={q.isLoading} error={q.error} />;

  const baseline = s.experiments.find(e => e.is_baseline)?.metrics.gini_weighted ?? null;
  const final = s.playoff?.final ? s.playoff.finalists.find(f => f.delegation_id === s.playoff!.final!.delegation_id) : null;
  const champion = s.experiments.find(e => e.experiment_id === (final?.experiment_id ?? s.champion_timeline.at(-1)?.new_champion_id));
  const totals = s.telemetry_summary.totals;
  const decided = s.delegations.reduce((n, d) => n + d.budget.decided, 0);
  const provisional = s.campaign.status === 'consolidating' && !s.playoff?.final;
  const championLabel = championDescriptor(s.experiments, champion, final?.model_family);

  return <main className="page overview">
    <section className="overview-hero panel">
      <div>
        <span className="eyebrow">{s.campaign.status === 'completed' ? 'Final champion' : provisional ? 'Provisional leader · playoff incomplete' : `Campaign leader · ${s.campaign.status}`}</span>
        <h1>{championLabel}</h1>
        {provisional && s.playoff?.failure_reason && <p className="playoff-failure">{s.playoff.failure_reason}</p>}
        <p className="mono">{final?.experiment_id ?? champion?.experiment_id}</p>
        <Link to="#playoff"><Badge kind={provisional ? 'takeover' : 'promote'}>{s.playoff?.decision_mode ?? 'no playoff'} verdict</Badge></Link>
      </div>
      <StatTile label="Final gini" value={<MetricNumber value={final?.gini_weighted ?? champion?.metrics.gini_weighted ?? null} />} />
      <StatTile label="vs baseline" value={<MetricNumber kind="lift" value={baseline == null ? null : (final?.gini_weighted ?? champion?.metrics.gini_weighted ?? 0) - baseline} />} />
      <StatTile label="Dataset" value={`${s.campaign.dataset} / ${s.campaign.target_mode}`} />
      <StatTile label="Duration" value={duration((new Date(s.campaign.ended_at ?? s.campaign.created_at).getTime() - new Date(s.campaign.created_at).getTime()) / 60000)} />
      <StatTile label="Orchestrator" value={orchestratorLabel(s.campaign.orchestrator, s.campaign.orchestrator_model)} />
    </section>
    <ChampionAscent orchId={s.campaign.orch_id} experiments={s.experiments} championTimeline={s.champion_timeline} delegations={s.delegations} notes={s.notes} />
    <div className="overview-grid">
      <section className="panel ledger"><header className="section-title"><div><span className="eyebrow">Budget ledger</span><h2>Cycles by delegation</h2></div></header>
        {s.delegations.map(d => {
          const used = Math.min(d.budget.decided, d.budget.committed);
          const unused = Math.max(0, d.budget.committed - used - d.budget.forfeited);
          return <Link className="budget-row" key={d.delegation_id} to={`/o/${orchId}/delegations/${d.delegation_id}`} title={`${d.budget.attempted} attempts · ${s.experiments.filter(e => e.delegation_id === d.delegation_id && e.is_seed).length} seed evaluations`}><b>{d.delegation_id}</b><div className="budget-track"><i className="used" style={{ flex: used }} /><i className="forfeited" style={{ flex: d.budget.forfeited }} /><i className="unused" style={{ flex: unused }} /></div><span>{used}/{d.budget.committed}</span></Link>;
        })}
      </section>
      <section className="panel cost-panel"><header className="section-title"><div><span className="eyebrow">Cost panel</span><h2>{compact(totalTokens(totals))} tokens</h2></div><Link to={`/o/${orchId}/telemetry`}>Open analytics →</Link></header><div className="cost-stack"><i className="cached" style={{ flex: totals.cached_input }} /><i className="uncached" style={{ flex: Math.max(0, totals.input - totals.cached_input) }} /><i className="output" style={{ flex: totals.output }} /><i className="reasoning" style={{ flex: totals.reasoning }} /></div><div className="stat-grid"><StatTile label="Cache hit" value={<MetricNumber kind="percent" value={s.telemetry_summary.cache_hit_rate} />} /><StatTile label="Tokens / decision" value={compact(totalTokens(totals) / Math.max(decided, 1))} /><StatTile label="Model calls" value={totals.model_calls} /><StatTile label="Tool calls" value={totals.tool_calls} /></div></section>
    </div>
    <section className="distress-section"><header className="section-title"><div><span className="eyebrow">Distress board</span><h2>Raised flags</h2></div></header><div className="distress-grid">{s.delegations.flatMap(d => [...new Set(d.distress.active)].map(flag => <Link className="distress-card panel" key={`${d.delegation_id}-${flag}`} to={`/o/${orchId}/journey#chapter-${d.delegation_id}`}><Badge kind={flag === 'early_stop' ? 'promote' : flag === 'auto_rejected' ? 'reject' : 'distress'}>{flag}</Badge><b>{d.delegation_id}</b><p>{DISTRESS_COPY[flag] ?? d.distress.detail ?? 'Flag raised during delegation.'}</p>{d.taken_over && <Badge kind="takeover">takeover recovered</Badge>}</Link>))}</div></section>
    <ResearchTree orchId={s.campaign.orch_id} experiments={s.experiments} delegations={s.delegations} />
    <section className="panel playoff-panel" id="playoff"><header className="section-title"><div><span className="eyebrow">Playoff · {s.playoff?.status ?? 'unavailable'}</span><h2>Final consolidation</h2></div><Badge kind="orchestrator">{s.playoff?.decision_mode ?? 'unavailable'}</Badge></header>
      {s.playoff ? <>
        <div className="playoff-table">{s.playoff.finalists.map(f => { const e=s.experiments.find(x=>x.experiment_id===f.experiment_id); return <Link key={f.delegation_id} to={`/o/${orchId}/delegations/${f.delegation_id}`}><b>{f.delegation_id}</b><span>{championDescriptor(s.experiments,e,f.model_family)}</span><MetricNumber value={f.gini_weighted} /></Link>; })}</div>
        {s.playoff.pairings.map(p => <div className="playoff-pairing" key={p.order}><b>Pairing {p.order} · {p.delegation_id}</b><Badge kind={p.decision === 'promote' ? 'promote' : 'reject'}>{p.decision ?? 'incomplete'}</Badge><span><MetricNumber kind="lift" value={p.mean_lift} /> · {Object.entries(p.gates).map(([gate,pass])=><i key={gate} className={pass?'gate-pass':'gate-fail'}>{pass?'✓':'×'} {gate}</i>)}</span></div>)}
        {s.playoff.exclusions.map(e => <p className="playoff-exclusion" key={`${e.delegation_id}-${e.reason}`}>Excluded {e.delegation_id ?? 'finalist'} · {e.reason}</p>)}
        {s.playoff.final && <p className="lineage mono">{s.playoff.final.source_experiment_id} → {s.playoff.final.consolidation_experiment_id}</p>}
        {!s.playoff.final && s.playoff.failure_reason && <p className="playoff-failure">{s.playoff.failure_reason}</p>}
        {s.playoff.report_md && <Link to={`/o/${orchId}/files/${s.playoff.report_md}`}>Open playoff evidence →</Link>}
      </> : <p>No playoff record.</p>}
    </section>
  </main>;
}
