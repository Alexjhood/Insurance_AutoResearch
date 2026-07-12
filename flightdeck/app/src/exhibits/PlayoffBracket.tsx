import { useMemo, useState } from 'react';
import { ChevronUp, RotateCcw } from 'lucide-react';
import { Badge } from '../components/Badge';
import { MetricNumber } from '../components/MetricNumber';
import type { Experiment, Playoff } from '../lib/types';
import { championDescriptor } from '../lib/format';
import { delegationVar } from './shared';
import './exhibits.css';

interface PlayoffBracketProps {
  playoff: Playoff | null;
  experiments: Experiment[];
  compact?: boolean;
}

function shortId(value: string | null | undefined) {
  if (!value) return 'unresolved';
  return value.replace(/^\d{8}T\d{6}Z_/, '').replace(/^orchestration_(challenger|seed)_\d+_/, '');
}

function gateLabel(value: string) {
  return value.replaceAll('_', ' ');
}

export function PlayoffBracket({ playoff, experiments, compact = false }: PlayoffBracketProps) {
  const [flipped, setFlipped] = useState<Set<number>>(new Set());
  const lineage = useMemo(() => {
    if (!playoff?.finalists.length) return [];
    let incumbent = playoff.finalists[0];
    const steps = [{ delegationId: incumbent.delegation_id, experimentId: incumbent.experiment_id, order: 0 }];
    for (const pairing of playoff.pairings) {
      if (pairing.decision === 'promote') {
        const challenger = playoff.finalists.find(f => f.delegation_id === pairing.delegation_id);
        if (challenger) incumbent = challenger;
        steps.push({ delegationId: pairing.delegation_id, experimentId: pairing.source_experiment_id, order: pairing.order });
      }
    }
    return steps;
  }, [playoff]);

  if (!playoff) return <section className="playoff-bracket empty panel"><Badge kind="takeover">Playoff unavailable</Badge><p>No playoff evidence was recorded for this campaign.</p></section>;

  const toggle = (order: number) => setFlipped(current => {
    const next = new Set(current);
    if (next.has(order)) next.delete(order); else next.add(order);
    return next;
  });
  const complete = playoff.status === 'completed' && Boolean(playoff.final);

  return <section className={`playoff-bracket panel ${compact ? 'compact' : ''}`} aria-label="Ascending playoff gauntlet">
    <header className="section-title">
      <div><span className="eyebrow">Playoff bracket · ascending gauntlet</span><h2>{complete ? 'The climb to campaign champion' : 'Playoff evidence to date'}</h2></div>
      <Badge kind={complete ? 'promote' : 'takeover'}>{complete ? 'complete' : playoff.status || 'incomplete'}</Badge>
    </header>

    <div className="playoff-lineage" aria-label="Champion lineage">
      <span className="lineage-label">Champion lineage</span>
      {lineage.map((step, index) => <div className="lineage-step" key={`${step.order}-${step.delegationId}`} style={{ color: `var(${delegationVar(step.delegationId)})` }}>
        {index > 0 && <ChevronUp aria-hidden="true" />}
        <b>{step.delegationId}</b><span>{shortId(step.experimentId)}</span>
      </div>)}
      {!complete && <div className="lineage-pending"><span>…</span> crown unresolved</div>}
    </div>

    <div className="gauntlet">
      {playoff.pairings.map(pairing => {
        const finalist = playoff.finalists.find(f => f.delegation_id === pairing.delegation_id);
        const experiment = experiments.find(e => e.experiment_id === pairing.source_experiment_id);
        const isFlipped = flipped.has(pairing.order);
        const passed = pairing.decision === 'promote';
        const decided = pairing.decision != null;
        return <button type="button" className={`pairing-card ${isFlipped ? 'flipped' : ''}`} key={pairing.order} onClick={() => toggle(pairing.order)} aria-pressed={isFlipped} aria-label={`Pairing ${pairing.order}, ${pairing.delegation_id}, ${pairing.decision ?? 'incomplete'}; ${isFlipped ? 'show summary' : 'show gate evidence'}`}>
          <span className="pairing-rung">{String(pairing.order).padStart(2, '0')}</span>
          <span className="pairing-inner">
            <span className="pairing-face pairing-front">
              <span className="pairing-top"><i style={{ background: `var(${delegationVar(pairing.delegation_id)})` }} /><b>{pairing.delegation_id}</b><Badge kind={passed ? 'promote' : decided ? 'reject' : 'takeover'}>{pairing.decision ?? 'incomplete'}</Badge></span>
              <strong>{championDescriptor(experiments, experiment, finalist?.model_family)}</strong>
              <span className="pairing-metrics"><span>replayed gini <MetricNumber value={finalist?.gini_weighted ?? null} /></span><span>CV lift <MetricNumber kind="lift" value={pairing.mean_lift} /></span><span>win rate <MetricNumber kind="percent" value={pairing.challenger_win_rate} /></span></span>
              <small>{pairing.decision_reason ?? 'Awaiting comparison evidence.'}</small>
              <span className="flip-hint"><RotateCcw aria-hidden="true" /> gate evidence</span>
            </span>
            <span className="pairing-face pairing-back">
              <span className="pairing-top"><b>{pairing.delegation_id} gate grid</b><Badge kind={passed ? 'promote' : decided ? 'reject' : 'takeover'}>{passed ? 'advanced' : decided ? 'held' : 'pending'}</Badge></span>
              <span className="gate-grid">{Object.entries(pairing.gates).map(([gate, ok]) => <span className={ok ? 'pass' : 'fail'} key={gate}><i>{ok ? '✓' : '×'}</i>{gateLabel(gate)}</span>)}</span>
              <span className={`guardrail-chip ${pairing.guardrail_passed ? 'pass' : 'fail'}`}>{pairing.guardrail_passed ? '✓' : '×'} hard guardrail</span>
              <span className="flip-hint"><RotateCcw aria-hidden="true" /> pairing summary</span>
            </span>
          </span>
        </button>;
      })}
      {playoff.pairings.length === 0 && <div className="playoff-incomplete"><Badge kind="takeover">No pairings completed</Badge><p>The field was assembled, but the gauntlet did not begin.</p></div>}
    </div>

    {playoff.exclusions.length > 0 && <aside className="playoff-bench"><span className="eyebrow">Excluded from playoff</span>{playoff.exclusions.map((item, index) => <div key={`${item.delegation_id}-${index}`}><Badge kind="distress">{item.delegation_id ?? 'finalist'} excluded</Badge><span>{gateLabel(item.reason)}</span></div>)}</aside>}
    {!complete && playoff.failure_reason && <div className="playoff-incomplete"><Badge kind="distress">Playoff incomplete</Badge><p>{playoff.failure_reason}</p></div>}
  </section>;
}
