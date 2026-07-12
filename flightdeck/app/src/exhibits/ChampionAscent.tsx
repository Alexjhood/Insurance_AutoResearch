import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { scaleLinear } from 'd3-scale';
import type { ChampionEvent, Delegation, Experiment, OperatorNote } from '../lib/types';
import {
  ExhibitShell, LegendItem, SegmentToggle, useExhibitTooltip, useMeasure, useReducedMotion,
  outcomeOf, OUTCOME_GLYPH, OUTCOME_LABEL, OUTCOME_VAR, delegationVar, parseTime, clockLabel,
} from './shared';
import './exhibits.css';

export interface ChampionAscentProps { orchId: string; experiments: Experiment[]; championTimeline: ChampionEvent[]; delegations: Delegation[]; notes: OperatorNote[] }

const HEIGHT = 380;
const M = { l: 66, r: 18, t: 18, b: 46 };

/** Journey anchor for an experiment (matches Journey card / chapter ids). */
function journeyHash(e: Experiment): string {
  if (e.delegation_id == null) return '#finale';
  if (e.is_seed || e.is_baseline) return `#chapter-${e.delegation_id}`;
  return `#${e.delegation_id}-x${e.cycle ?? e.seq}`;
}

export function ChampionAscent({ orchId, experiments, championTimeline, delegations, notes }: ChampionAscentProps) {
  const [wrapRef, width] = useMeasure<HTMLDivElement>();
  const [scaleMode, setScaleMode] = useState<'magnify' | 'full'>('magnify');
  const { tooltip, show, hide } = useExhibitTooltip();
  const navigate = useNavigate();
  const reduced = useReducedMotion();
  const w = Math.max(width, 640);

  const model = useMemo(() => {
    const maxSeq = Math.max(...experiments.map(e => e.seq), 1);
    const x = scaleLinear([-0.5, maxSeq + 0.5], [M.l, w - M.r]);

    const ginis = experiments.map(e => e.metrics.gini_weighted).filter((g): g is number => g != null);
    const top = Math.max(...ginis, 0);
    // Magnified domain: the champion's neighborhood — everything within 0.015 of the peak.
    const near = ginis.filter(g => g >= top - 0.015);
    const magMin = Math.min(...near) - 0.0012, magMax = top + 0.0012;
    const domain: [number, number] = scaleMode === 'magnify' ? [magMin, magMax] : [0, top * 1.06];
    const y = scaleLinear(domain, [HEIGHT - M.b, M.t]);

    // Delegation bands (+ consolidation band for delegation_id null)
    const bandDefs: { id: string; label: string; colorVar: string }[] = [
      ...delegations.map(d => ({ id: d.delegation_id, label: d.delegation_id, colorVar: delegationVar(d.delegation_id) })),
      { id: '__playoff', label: 'playoff', colorVar: delegationVar(null) },
    ];
    const bands = bandDefs.flatMap(b => {
      const es = experiments.filter(e => (e.delegation_id ?? '__playoff') === b.id);
      if (!es.length) return [];
      const s0 = Math.min(...es.map(e => e.seq)), s1 = Math.max(...es.map(e => e.seq));
      const at = parseTime(es[0].created_at);
      return [{ ...b, x0: x(s0 - 0.5), x1: x(s1 + 0.5), clock: at == null ? null : clockLabel(at) }];
    });

    // Champion step line: moves only on champion events, mapped to the event experiment's seq.
    const seqOf = (id: string) => experiments.find(e => e.experiment_id === id)?.seq;
    const steps: [number, number][] = [];
    for (const ev of championTimeline) {
      if (ev.new_champion_gini == null) continue;
      const s = seqOf(ev.new_champion_id);
      if (s == null) continue;
      steps.push([s, ev.new_champion_gini]);
    }
    steps.sort((a, b) => a[0] - b[0]);
    let path = '';
    steps.forEach(([s, g], i) => {
      const px = x(s), py = y(Math.max(domain[0], Math.min(domain[1], g)));
      if (i === 0) path += `M ${px} ${py}`;
      else path += ` L ${px} ${steps[i - 1] ? y(Math.max(domain[0], Math.min(domain[1], steps[i - 1][1]))) : py} L ${px} ${py}`;
    });
    if (steps.length) {
      const last = steps[steps.length - 1];
      path += ` L ${x(maxSeq + 0.5)} ${y(Math.max(domain[0], Math.min(domain[1], last[1])))}`;
    }

    // Vertical event rules: takeover notes (amber), crashed/unclean delegations (red)
    const rules: { x: number; colorVar: string; glyph: string; label: string }[] = [];
    for (const n of notes.filter(n => n.kind === 'takeover' && n.delegation_id)) {
      const es = experiments.filter(e => e.delegation_id === n.delegation_id);
      if (!es.length) continue;
      rules.push({ x: x(Math.max(...es.map(e => e.seq)) + 0.5), colorVar: '--c-takeover', glyph: '⚑', label: `takeover · ${n.delegation_id}` });
    }
    for (const d of delegations.filter(d => d.clean_exit === false || d.distress.active.includes('crashed'))) {
      const es = experiments.filter(e => e.delegation_id === d.delegation_id);
      if (!es.length) continue;
      rules.push({ x: x(Math.min(...es.map(e => e.seq)) - 0.5), colorVar: '--c-distress', glyph: '✕', label: `distress · ${d.delegation_id}` });
    }

    const ticks = y.ticks(scaleMode === 'magnify' ? 6 : 5);
    return { x, y, domain, bands, path, rules, ticks, maxSeq };
  }, [experiments, championTimeline, delegations, notes, w, scaleMode]);

  const finalEvent = championTimeline[championTimeline.length - 1];
  const belowDomain = experiments.filter(e => e.metrics.gini_weighted != null && e.metrics.gini_weighted < model.domain[0]);
  const labelStride = Math.max(1, Math.ceil(belowDomain.length / Math.max(1, Math.floor(w / 120))));

  const tipFor = (e: Experiment) => {
    const o = outcomeOf(e);
    return (
      <>
        <div className="tip-title">{e.name}</div>
        <div className="tip-meta mono">
          {e.delegation_id ?? 'playoff'}{e.cycle != null ? ` · cycle ${e.cycle}` : ''} · {e.model_family}
          {e.metrics.gini_weighted != null && <> · gini {e.metrics.gini_weighted.toFixed(4)}</>}
        </div>
        <div className="tip-meta mono">
          <span style={{ color: `var(${OUTCOME_VAR[o]})` }}>{OUTCOME_GLYPH[o]} {OUTCOME_LABEL[o]}</span>
          {e.comparison?.decision_reason_code && <> · {e.comparison.decision_reason_code}</>}
          {e.lift.vs_then_champion != null && <> · lift {e.lift.vs_then_champion >= 0 ? '+' : '−'}{Math.abs(e.lift.vs_then_champion).toFixed(4)}</>}
          {e.comparison?.fold_win_rate != null && <> · {(e.comparison.fold_win_rate * 100).toFixed(0)}% folds</>}
        </div>
        {e.recipe != null && typeof e.recipe === 'object' && 'estimator' in (e.recipe as Record<string, unknown>) && (
          <div className="tip-meta mono">{String((e.recipe as Record<string, unknown>).estimator)} · {String((e.recipe as Record<string, unknown>).objective ?? '')} · {String((e.recipe as Record<string, unknown>).encoding ?? '')}</div>
        )}
        <p className="tip-body">{e.comparison?.decision_rationale ?? e.proposal.hypothesis ?? ''}</p>
        <div className="tip-hint">click → journey card</div>
      </>
    );
  };

  const table = (
    <>
      <thead><tr><th>#</th><th>Experiment</th><th>Delegation</th><th>Cycle</th><th>Decision</th><th className="num">Gini</th><th className="num">Lift vs champion</th><th className="num">Fold wins</th></tr></thead>
      <tbody>
        {experiments.map(e => {
          const o = outcomeOf(e);
          return (
            <tr key={e.experiment_id}>
              <td className="num">{e.seq}</td><td className="mono">{e.name}</td>
              <td><span style={{ color: `var(${delegationVar(e.delegation_id)})` }}>{e.delegation_id ?? 'playoff'}</span></td>
              <td className="num">{e.cycle ?? '—'}</td>
              <td><span style={{ color: `var(${OUTCOME_VAR[o]})` }}>{OUTCOME_GLYPH[o]} {OUTCOME_LABEL[o]}</span></td>
              <td className="num">{e.metrics.gini_weighted?.toFixed(4) ?? '—'}</td>
              <td className="num">{e.lift.vs_then_champion == null ? '—' : `${e.lift.vs_then_champion >= 0 ? '+' : '−'}${Math.abs(e.lift.vs_then_champion).toFixed(4)}`}</td>
              <td className="num">{e.comparison?.fold_win_rate == null ? '—' : `${(e.comparison.fold_win_rate * 100).toFixed(0)}%`}</td>
            </tr>
          );
        })}
      </tbody>
    </>
  );

  return (
    <ExhibitShell
      eyebrow="Champion ascent" title="The search trajectory"
      note={`${experiments.filter(e => !e.is_seed && !e.is_baseline).length} experiments · ${delegations.length} delegations`}
      controls={<SegmentToggle label="y-scale" value={scaleMode} onChange={(v: 'magnify' | 'full') => setScaleMode(v)}
        options={[{ value: 'magnify', label: 'Magnify the ascent' }, { value: 'full', label: 'Full scale' }]} />}
      legend={<>
        <LegendItem colorVar="--c-promote" glyph="▲">promote</LegendItem>
        <LegendItem colorVar="--c-localpromote" glyph="◭">local promote</LegendItem>
        <LegendItem colorVar="--c-reject" glyph="○">reject</LegendItem>
        <LegendItem colorVar="--c-distress" glyph="✕">forfeited / undecided</LegendItem>
        <LegendItem colorVar="--c-baseline" glyph="⇢">seed transfer</LegendItem>
        <LegendItem colorVar="--c-orchestrator" glyph="◈">playoff replay</LegendItem>
        <LegendItem colorVar="--c-takeover" glyph="⚑">takeover</LegendItem>
        <LegendItem colorVar="--c-promote" glyph="—">champion line</LegendItem>
      </>}
      table={table} tableCaption="Every campaign experiment with decision, gini and lift"
    >
      <div className="chart-well" ref={wrapRef}>
        <svg className={`ascent-svg ${reduced ? '' : 'animate-in'}`} width="100%" height={HEIGHT} viewBox={`0 0 ${w} ${HEIGHT}`} role="img"
          aria-label="Champion ascent: weighted gini of every experiment over the campaign, with the champion step line">
          {/* delegation bands */}
          {model.bands.map(b => (
            <g key={b.id}>
              <rect className="band" x={b.x0} y={M.t} width={Math.max(2, b.x1 - b.x0)} height={HEIGHT - M.t - M.b} style={{ fill: `var(${b.colorVar})` }} />
              <text className="band-label" x={(b.x0 + b.x1) / 2} y={HEIGHT - 26} textAnchor="middle" style={{ fill: `var(${b.colorVar})` }}>{b.label}</text>
              {b.clock && <text className="axis-label" x={(b.x0 + b.x1) / 2} y={HEIGHT - 10} textAnchor="middle">{b.clock}</text>}
            </g>
          ))}
          {/* y grid */}
          {model.ticks.map(t => (
            <g key={t}>
              <line className="grid" x1={M.l} x2={w - M.r} y1={model.y(t)} y2={model.y(t)} />
              <text className="axis-label" x={M.l - 8} y={model.y(t) + 4} textAnchor="end">{t.toFixed(scaleMode === 'magnify' ? 4 : 2)}</text>
            </g>
          ))}
          <text className="axis-label" x={M.l - 8} y={M.t - 4} textAnchor="end">gini_w</text>
          {/* event rules */}
          {model.rules.map((r, i) => (
            <g key={i}>
              <line className="event-rule" x1={r.x} x2={r.x} y1={M.t} y2={HEIGHT - M.b} style={{ stroke: `var(${r.colorVar})` }} />
              <text className="rule-label" x={r.x + 5} y={M.t + 12} style={{ fill: `var(${r.colorVar})` }}>{r.glyph} {r.label}</text>
            </g>
          ))}
          {/* champion step line */}
          <path className="champion-step" d={model.path} pathLength={1} />
          {/* experiment points */}
          {experiments.map(e => {
            if (e.metrics.gini_weighted == null) return null;
            const o = outcomeOf(e);
            const clamped = e.metrics.gini_weighted < model.domain[0];
            const py = clamped ? HEIGHT - M.b - 10 : model.y(Math.min(e.metrics.gini_weighted, model.domain[1]));
            const px = model.x(e.seq);
            return (
              <g key={e.experiment_id} className="point-hit" tabIndex={0} role="link"
                aria-label={`${e.name}, ${OUTCOME_LABEL[o]}, gini ${e.metrics.gini_weighted.toFixed(4)} — open journey card`}
                onMouseMove={ev => show(ev, tipFor(e))} onMouseLeave={hide}
                onFocus={() => show({ clientX: 80, clientY: 120 }, tipFor(e))} onBlur={hide}
                onClick={() => { hide(); navigate(`/o/${orchId}/journey${journeyHash(e)}`); }}
                onKeyDown={ev => { if (ev.key === 'Enter') { hide(); navigate(`/o/${orchId}/journey${journeyHash(e)}`); } }}>
                <circle cx={px} cy={py} r={12} fill="transparent" />
                {clamped && belowDomain.indexOf(e) % labelStride === 0 && <text className="clamp-label" x={px} y={HEIGHT - M.b - 24 - (Math.floor(belowDomain.indexOf(e) / labelStride) % 2) * 13} textAnchor="middle">▼ {e.metrics.gini_weighted.toFixed(4)}</text>}
                <text className={`point-glyph ${o === 'seed' ? 'seed' : ''}`} x={px} y={py + 5} textAnchor="middle"
                  style={{ fill: `var(${OUTCOME_VAR[o]})`, fontSize: o === 'promote' || o === 'playoff' ? 17 : 14 }}>{OUTCOME_GLYPH[o]}</text>
              </g>
            );
          })}
          {/* final champion annotation */}
          {finalEvent?.new_champion_gini != null && (
            <text className="final-label" x={w - M.r - 4} y={model.y(Math.min(finalEvent.new_champion_gini, model.domain[1])) - 12} textAnchor="end">
              {finalEvent.scope === 'consolidation' && finalEvent.is_seed_transfer ? 'consolidation champion' : 'final champion'} {finalEvent.new_champion_gini.toFixed(4)}
            </text>
          )}
        </svg>
        {tooltip}
      </div>
    </ExhibitShell>
  );
}
