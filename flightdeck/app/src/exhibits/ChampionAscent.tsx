import { Link } from 'react-router-dom';
import type { ChampionEvent, Delegation, Experiment, OperatorNote } from '../lib/types';
import './exhibits.css';

export interface ChampionAscentProps { orchId: string; experiments: Experiment[]; championTimeline: ChampionEvent[]; delegations: Delegation[]; notes: OperatorNote[] }
export function ChampionAscent({ orchId, experiments, championTimeline, delegations }: ChampionAscentProps) {
  const points = experiments.filter(e => e.metrics.gini_weighted != null); const values = points.map(e => e.metrics.gini_weighted!);
  const min = Math.min(...values), max = Math.max(...values), range = max - min || .01;
  const x = (seq: number) => 42 + (seq / Math.max(experiments.length - 1, 1)) * 818; const y = (v: number) => 224 - ((v - min) / range) * 178;
  return <section className="exhibit panel"><header className="section-title"><div><span className="eyebrow">Champion ascent</span><h2>The search trajectory</h2></div><span>{championTimeline.length} champion events</span></header>
    <svg className="ascent-chart" viewBox="0 0 900 260" role="img" aria-label="Experiment gini and champion progression">
      {[0,.25,.5,.75,1].map(t => <line key={t} x1="42" x2="870" y1={224-t*178} y2={224-t*178} />)}
      {delegations.map((d,i) => { const es=experiments.filter(e=>e.delegation_id===d.delegation_id); if(!es.length)return null; return <rect key={d.delegation_id} className={`delegation-band d${i+1}`} x={x(es[0].seq)-10} y="24" width={Math.max(18,x(es.at(-1)!.seq)-x(es[0].seq)+20)} height="205"><title>{d.delegation_id}</title></rect>; })}
      {points.map(e => <Link key={e.experiment_id} to={`/o/${orchId}/journey#${e.delegation_id ?? 'playoff'}-x${e.cycle ?? e.seq}`}><circle className={`point decision-${e.comparison?.decision ?? 'pending'}`} cx={x(e.seq)} cy={y(e.metrics.gini_weighted!)} r="5"><title>{e.name}: {e.metrics.gini_weighted!.toFixed(4)}</title></circle></Link>)}
      <polyline className="champion-line" points={championTimeline.filter(e=>e.new_champion_gini!=null).map((e,i)=>`${x(experiments.find(x=>x.experiment_id===e.new_champion_id)?.seq ?? i)},${y(e.new_champion_gini!)}`).join(' ')} />
      <text x="42" y="250">launch</text><text x="820" y="250">playoff</text>
    </svg>
  </section>;
}
