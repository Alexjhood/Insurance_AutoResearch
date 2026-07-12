import { useEffect, useState } from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';
import { Badge } from '../../components/Badge';
import { MetricNumber } from '../../components/MetricNumber';
import { PageState } from '../../components/PageState';
import { MissionTimeline } from '../../exhibits/MissionTimeline';
import { PlayoffBracket } from '../../exhibits/PlayoffBracket';
import { decisionKind, recipeDiff } from '../../lib/format';
import { useSnapshot } from '../../lib/data/queries';
import type { Experiment } from '../../lib/types';
import { useExperimentHover } from '../../lib/experimentHover';
import './Journey.css';
import './RecipeDiff.css';
import './JourneyRail.css';

function ExperimentCard({ e, experiments, orchId, forfeited }: { e: Experiment; experiments: Experiment[]; orchId: string; forfeited: boolean }) {
  const decision = e.comparison?.decision;
  const diffs = recipeDiff(experiments, e);
  const { showExperiment, moveExperiment, clearExperiment, classNameFor } = useExperimentHover();
  return <article className={`experiment-card panel ${classNameFor(e.experiment_id)}`} id={`${e.delegation_id ?? 'playoff'}-x${e.cycle ?? e.seq}`} onMouseEnter={event => showExperiment(event, e)} onMouseMove={moveExperiment} onMouseLeave={clearExperiment}>
    <header><div><span className="eyebrow">Cycle {e.cycle ?? 'seed'} · {e.model_family}</span><h3>{e.name}</h3></div><Badge kind={forfeited ? 'distress' : decisionKind(decision, e.status)}>{forfeited ? 'forfeited' : decision ?? (e.is_seed ? 'seed' : 'pending')}</Badge></header>
    <p className="hypothesis">{e.proposal.hypothesis ?? 'Seed or framework-generated experiment.'}</p>
    {diffs.length > 0 && <div className="recipe-diffs" aria-label="Recipe changes">{diffs.slice(0, 4).map(diff => <span key={diff}>{diff}</span>)}{diffs.length > 4 && <span>+{diffs.length - 4} more</span>}</div>}
    {e.proposal.change_summary && <p><b>Changed:</b> {e.proposal.change_summary}</p>}
    <div className="experiment-metrics"><span>Gini <MetricNumber value={e.metrics.gini_weighted} /></span><span>Lift <MetricNumber kind="lift" value={e.lift.vs_then_champion} /></span><span>Win rate <MetricNumber kind="percent" value={e.comparison?.fold_win_rate ?? e.screening?.win_rate ?? null} /></span><span>Calibration <MetricNumber value={e.metrics.calibration_ratio} /></span><span>Pricing loss <MetricNumber value={e.metrics.asym_pricing_loss} /></span></div>
    {e.comparison?.decision_reason_code && <p className="decision-copy"><b>{e.comparison.decision_reason_code}</b> · {e.comparison.decision_rationale}</p>}
    {e.decision_meta.interpretation && <blockquote>{e.decision_meta.interpretation}</blockquote>}
    {e.repairs.map(r => <div className="repair-card" key={r.attempt}><Badge kind="takeover">repair {r.attempt}</Badge> {r.failed_checks.join(', ')} · {r.resolved ? 'resolved' : 'failed'}</div>)}
    <footer><Link to={`/o/${orchId}/delegations/${e.delegation_id}`}>Delegation evidence →</Link></footer>
  </article>;
}

export function Journey() {
  const { orchId } = useParams();
  const q = useSnapshot(orchId);
  const s = q.data;
  const [activeChapter, setActiveChapter] = useState<string | null>(null);
  const { hash } = useLocation();
  useEffect(() => {
    if (!s || !hash) return;
    const el = document.getElementById(hash.slice(1));
    if (el) requestAnimationFrame(() => el.scrollIntoView({ block: 'center' }));
  }, [s, hash]);
  useEffect(() => {
    if (!s) return;
    const chapters = Array.from(document.querySelectorAll<HTMLElement>('[data-journey-chapter]'));
    const observer = new IntersectionObserver(entries => entries.forEach(entry => entry.isIntersecting && setActiveChapter(entry.target.id)), { rootMargin: '-35% 0px -55% 0px' });
    chapters.forEach(c => observer.observe(c));
    return () => observer.disconnect();
  }, [s]);
  if (!s) return <PageState loading={q.isLoading} error={q.error} />;
  const plan = s.notes.find(n => !n.delegation_id && (n.kind === 'plan' || n.kind === 'decision'));
  return <main className="page journey">
    <nav className="journey-rail" aria-label="Journey chapters"><a href="#launch" className={activeChapter === 'launch' ? 'active' : ''}><i />Launch</a>{s.delegations.map(delegation => <a href={`#${delegation.delegation_id}`} className={activeChapter === delegation.delegation_id ? 'active' : ''} key={delegation.delegation_id}><i />{delegation.delegation_id}</a>)}<a href="#finale" className={activeChapter === 'finale' ? 'active' : ''}><i />Finale</a></nav>
    <header className="journey-heading"><span className="eyebrow">Campaign narrative</span><h1>The Journey</h1><p>From global mean to consolidated champion, with every decision linked to its evidence.</p></header>
    <MissionTimeline delegations={s.delegations} experiments={s.experiments} notes={s.notes} activeDelegationId={activeChapter} />
    <div className="narrative">
      <section className="launch panel" id="launch" data-journey-chapter><Badge kind="baseline">Launch</Badge><h2>{s.campaign.dataset} / {s.campaign.target_mode}</h2>{plan && <blockquote><b>Orchestrator plan</b><br />{plan.text}</blockquote>}<p>{s.delegations[0]?.brief.direction}</p><div className="chips">{s.delegations[0]?.brief.starting_knowledge.map(x => <span key={x}>{x}</span>)}</div></section>
      {s.delegations.map((d, di) => {
        const experiments = s.experiments.filter(e => e.delegation_id === d.delegation_id);
        const notes = s.notes.filter(n => n.delegation_id === d.delegation_id && (n.kind === 'reflection' || n.kind === 'takeover')).sort((a, b) => a.at.localeCompare(b.at));
        const label = (d.brief.name ?? d.backend).replace(new RegExp(`^${d.delegation_id}[_ ·-]*`, 'i'), '');
        return <div key={d.delegation_id}><section className="chapter" id={d.delegation_id} data-journey-chapter><header className="chapter-header"><span className="chapter-number">0{di + 1}</span><div><span className="eyebrow">Delegation chapter · {d.delegation_id}</span><h2>{label}</h2><p>{d.brief.direction}</p></div><Link to={`/o/${orchId}/delegations/${d.delegation_id}`}>Forensic view →</Link></header><div className="brief-card panel"><b>Mission brief</b><div className="chips">{d.brief.constraints.map(c => <span key={c}>✓ {c}</span>)}</div></div>{experiments.filter(e => !e.is_seed && !e.is_baseline).map(e => <ExperimentCard key={e.experiment_id} e={e} experiments={s.experiments} orchId={orchId!} forfeited={!e.comparison?.decision && d.ended_at != null} />)}</section>{notes.map((n, i) => <aside className={n.kind === 'takeover' ? 'takeover-scene panel' : 'interlude panel'} key={`${n.at}-${i}`}><Badge kind={n.kind === 'takeover' ? 'takeover' : 'reflection'}>{n.kind} · {d.delegation_id}</Badge><p>{n.text}</p></aside>)}</div>;
      })}
      <section className="finale" id="finale" data-journey-chapter><PlayoffBracket playoff={s.playoff} experiments={s.experiments} compact />{s.notes.filter(n => !n.delegation_id && n !== plan).map((n, i) => <blockquote className="finale-note panel" key={`${n.at}-${i}`}><Badge kind="reflection">Orchestrator {n.kind}</Badge><p>{n.text}</p></blockquote>)}<Link to={`/o/${orchId}#playoff`}>Open Overview playoff →</Link></section>
    </div>
  </main>;
}
