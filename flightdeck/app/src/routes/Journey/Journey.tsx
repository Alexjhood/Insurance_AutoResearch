import { useEffect, useState } from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';
import { Badge } from '../../components/Badge';
import { MetricNumber } from '../../components/MetricNumber';
import { PageState } from '../../components/PageState';
import { MissionTimeline } from '../../exhibits/MissionTimeline';
import { decisionKind, recipeSummary, resolveRecipe } from '../../lib/format';
import { useSnapshot } from '../../lib/data/queries';
import type { Experiment } from '../../lib/types';
import './Journey.css';

function ExperimentCard({e,orchId,forfeited}:{e:Experiment;orchId:string;forfeited:boolean}) { const decision=e.comparison?.decision; return <article className="experiment-card panel" id={`${e.delegation_id??'playoff'}-x${e.cycle??e.seq}`}><header><div><span className="eyebrow">Cycle {e.cycle??'seed'} · {e.model_family}</span><h3>{e.name}</h3></div><Badge kind={forfeited?'distress':decisionKind(decision,e.status)}>{forfeited?'forfeited':decision??(e.is_seed?'seed':'pending')}</Badge></header><p className="hypothesis">{e.proposal.hypothesis??'Seed or framework-generated experiment.'}</p>{e.proposal.change_summary&&<p><b>Changed:</b> {e.proposal.change_summary}</p>}<div className="experiment-metrics"><span>Gini <MetricNumber value={e.metrics.gini_weighted}/></span><span>Lift <MetricNumber kind="lift" value={e.lift.vs_then_champion}/></span><span>Win rate <MetricNumber kind="percent" value={e.comparison?.fold_win_rate??e.screening?.win_rate??null}/></span><span>Calibration <MetricNumber value={e.metrics.calibration_ratio}/></span><span>Pricing loss <MetricNumber value={e.metrics.asym_pricing_loss}/></span></div>{e.comparison?.decision_reason_code&&<p className="decision-copy"><b>{e.comparison.decision_reason_code}</b> · {e.comparison.decision_rationale}</p>}{e.decision_meta.interpretation&&<blockquote>{e.decision_meta.interpretation}</blockquote>}{e.repairs.map(r=><div className="repair-card" key={r.attempt}><Badge kind="takeover">repair {r.attempt}</Badge> {r.failed_checks.join(', ')} · {r.resolved?'resolved':'failed'}</div>)}<footer><Link to={`/o/${orchId}/delegations/${e.delegation_id}`}>Delegation evidence →</Link></footer></article> }
export function Journey(){const {orchId}=useParams();const q=useSnapshot(orchId);const s=q.data;
  // Scroll-sync: the chapter under the viewport centre drives the pinned MissionTimeline highlight.
  const [activeChapter,setActiveChapter]=useState<string|null>(null);
  // React-router doesn't scroll to #hash anchors on SPA navigation — do it once the data has rendered.
  const {hash}=useLocation();
  useEffect(()=>{
    if(!s||!hash)return;
    const el=document.getElementById(hash.slice(1));
    if(el)requestAnimationFrame(()=>{el.scrollIntoView({block:'center'});el.classList.add('deep-link-target');setTimeout(()=>el.classList.remove('deep-link-target'),1600);});
  },[s,hash]);
  useEffect(()=>{
    if(!s)return;
    const chapters=Array.from(document.querySelectorAll<HTMLElement>('section.chapter[id^="chapter-"]'));
    if(!chapters.length)return;
    const observer=new IntersectionObserver(entries=>{
      for(const entry of entries){ if(entry.isIntersecting){ setActiveChapter(entry.target.id.replace('chapter-','')); } }
    },{rootMargin:'-35% 0px -55% 0px'});
    chapters.forEach(c=>observer.observe(c));
    return ()=>observer.disconnect();
  },[s]);
  if(!s)return <PageState loading={q.isLoading} error={q.error}/>;return <main className="page journey"><header className="journey-heading"><span className="eyebrow">Campaign narrative</span><h1>The Journey</h1><p>From global mean to consolidated champion, with every decision linked to its evidence.</p></header><MissionTimeline delegations={s.delegations} experiments={s.experiments} notes={s.notes} activeDelegationId={activeChapter}/><div className="narrative"><section className="launch panel"><Badge kind="baseline">Launch</Badge><h2>{s.campaign.dataset} / {s.campaign.target_mode}</h2><p>{s.delegations[0]?.brief.direction}</p><div className="chips">{s.delegations[0]?.brief.starting_knowledge.map(x=><span key={x}>{x}</span>)}</div></section>{s.delegations.map((d,di)=>{const experiments=s.experiments.filter(e=>e.delegation_id===d.delegation_id);
  // The orchestrator writes takeover/reflection notes about a delegation after it ends, so they
  // render as interludes AFTER that delegation's chapter, in note-time order.
  const notes=s.notes.filter(n=>n.delegation_id===d.delegation_id&&(n.kind==='reflection'||n.kind==='takeover')).sort((a,b)=>a.at.localeCompare(b.at));
  return <div key={d.delegation_id}><section className="chapter" id={`chapter-${d.delegation_id}`}><header className="chapter-header"><span className="chapter-number">0{di+1}</span><div><span className="eyebrow">Delegation chapter</span><h2>{d.delegation_id} · {d.brief.name??d.backend}</h2><p>{d.brief.direction}</p></div><Link to={`/o/${orchId}/delegations/${d.delegation_id}`}>Forensic view →</Link></header><div className="brief-card panel"><b>Mission brief</b><div className="chips">{d.brief.constraints.map(c=><span key={c}>✓ {c}</span>)}</div>{d.brief.seed_champion&&<p className="mono">Seed: {d.brief.seed_champion.experiment_id}</p>}</div>{experiments.filter(e=>!e.is_seed&&!e.is_baseline).map(e=><ExperimentCard key={e.experiment_id} e={e} orchId={orchId!} forfeited={!e.comparison?.decision&&d.ended_at!=null}/>)}{d.distress.active.map(flag=><aside className="distress-callout" key={flag}><Badge kind="distress">{flag}</Badge><p>{d.distress.detail}</p></aside>)}</section>{notes.map((n,i)=>n.kind==='takeover'?<aside className="takeover-scene panel" key={`${n.at}-${i}`}><Badge kind="takeover">Takeover · {d.delegation_id}</Badge><h2>Mission recovery</h2><p>{n.text}</p></aside>:<aside className="interlude panel" key={`${n.at}-${i}`}><Badge kind="reflection">Orchestrator reflection · {d.delegation_id}</Badge><p>{n.text}</p></aside>)}</div>})}<section className="finale panel" id="finale"><Badge kind="promote">Finale</Badge><h2>Playoff consolidation</h2><div className="finalist-grid">{s.playoff?.finalists.map(f=><div key={f.delegation_id}><b>{f.delegation_id}</b><MetricNumber value={f.gini_weighted}/><span>{recipeSummary(resolveRecipe(s.experiments,s.experiments.find(e=>e.experiment_id===f.experiment_id)))??f.model_family}</span></div>)}</div>{s.notes.filter(n=>!n.delegation_id).map((n,i)=><blockquote className="finale-note" key={`${n.at}-${i}`}><Badge kind="reflection">Orchestrator {n.kind}</Badge><p>{n.text}</p></blockquote>)}{s.playoff?.final&&<p className="mono">Champion: {s.playoff.final.source_experiment_id}</p>}<Link to={`/o/${orchId}#playoff`}>Open playoff panel →</Link></section></div></main>}
