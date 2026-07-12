import { useMemo, useState } from 'react';
import { scaleLinear } from 'd3-scale';
import type { Delegation, Experiment, OperatorNote } from '../lib/types';
import { useExperimentHover } from '../lib/experimentHover';
import {
  LegendItem, useExhibitTooltip, useMeasure,
  outcomeOf, OUTCOME_LABEL, OUTCOME_VAR, delegationVar,
  parseTime, clockLabel, localTime, durationLabel,
} from './shared';
import './exhibits.css';

export interface MissionTimelineProps { delegations: Delegation[]; experiments: Experiment[]; notes: OperatorNote[]; activeDelegationId?: string | null }

const NOTE_GLYPH: Record<string, string> = { reflection: '◆', takeover: '⚑', decision: '▣' };
const NOTE_VAR: Record<string, string> = { reflection: '--c-orchestrator', takeover: '--c-takeover', decision: '--c-promote' };

function experimentHash(e: Experiment): string {
  if (e.delegation_id == null) return 'finale';
  if (e.is_seed || e.is_baseline) return e.delegation_id;
  return `${e.delegation_id}-x${e.cycle ?? e.seq}`;
}

export function MissionTimeline({ delegations, experiments, notes, activeDelegationId }: MissionTimelineProps) {
  const [wrapRef, width] = useMeasure<HTMLDivElement>();
  const [expanded, setExpanded] = useState(false);
  const { tooltip, show, hide } = useExhibitTooltip();
  const { showExperiment, moveExperiment, clearExperiment, classNameFor } = useExperimentHover();
  const w = Math.max(width, 640);
  const M = { l: 168, r: 20, t: 8, b: 20 };

  const scrollTo = (id: string) => {
    hide();
    const el = document.getElementById(id);
    if (el) {
      history.replaceState(null, '', `#${id}`);
      el.scrollIntoView({ block: 'center' });
      el.classList.add('deep-link-target');
      setTimeout(() => el.classList.remove('deep-link-target'), 1600);
    }
  };

  const model = useMemo(() => {
    const times: number[] = [];
    for (const d of delegations) { const a = parseTime(d.spawned_at), b = parseTime(d.ended_at); if (a) times.push(a); if (b) times.push(b); }
    for (const n of notes) { const t = parseTime(n.at); if (t) times.push(t); }
    for (const e of experiments) { const t = parseTime(e.created_at); if (t) times.push(t); }
    const t0 = Math.min(...times), t1 = Math.max(...times);
    const pad = (t1 - t0) * 0.015;
    const x = scaleLinear([t0 - pad, t1 + pad], [M.l, w - M.r]);

    const playoffExps = experiments.filter(e => e.delegation_id == null);
    const lanes: { id: string; label: string; colorVar: string; kind: 'orch' | 'delegation' | 'playoff' }[] = [
      { id: '__orch', label: 'orchestrator', colorVar: '--c-orchestrator', kind: 'orch' },
      ...delegations.map(d => {
        const name = d.brief.name ?? d.track;
        return { id: d.delegation_id, label: `${d.delegation_id} · ${name.length > 18 ? `${name.slice(0, 17)}…` : name}`, colorVar: delegationVar(d.delegation_id), kind: 'delegation' as const };
      }),
      ...(playoffExps.length ? [{ id: '__playoff', label: 'consolidation', colorVar: '--c-orchestrator', kind: 'playoff' as const }] : []),
    ];

    // Time ticks roughly every 10 minutes, capped to ~8
    const span = t1 - t0;
    const stepMs = Math.max(Math.ceil(span / 8 / 60000) * 60000, 60000);
    const ticks: number[] = [];
    for (let t = Math.ceil(t0 / stepMs) * stepMs; t <= t1; t += stepMs) ticks.push(t);

    return { x, lanes, ticks, t0, t1, playoffExps };
  }, [delegations, experiments, notes, w]);

  const height = (expanded ? 44 : 26) * model.lanes.length + M.t + M.b + 8;
  const laneY = (i: number) => M.t + 14 + i * (expanded ? 44 : 26);
  const orchY = laneY(0);
  const laneIndex = new Map(model.lanes.map((l, i) => [l.id, i]));

  const delegationTip = (d: Delegation) => (
    <>
      <div className="tip-title">{d.delegation_id} · {d.brief.name ?? d.backend}</div>
      <div className="tip-meta mono">{localTime(d.spawned_at)} → {localTime(d.ended_at)} · {durationLabel(d.cost.wall_clock_minutes == null ? null : d.cost.wall_clock_minutes * 60000)}</div>
      <div className="tip-meta mono">{d.budget.decided} decided / {d.budget.committed} committed{d.budget.forfeited ? ` · ${d.budget.forfeited} forfeited` : ''}{d.taken_over ? ' · taken over ⚑' : ''}</div>
      {d.distress.active.length > 0 && <div className="tip-meta" style={{ color: 'var(--c-distress)' }}>✕ {d.distress.active.join(', ')}</div>}
      <p className="tip-body">{d.brief.direction}</p>
      <div className="tip-hint">click → journey chapter</div>
    </>
  );

  return (
    <section className={`mission-timeline panel ${expanded ? 'expanded' : ''}`} aria-label="Mission timeline — campaign swimlane map">
      <div className="mission-head">
        <span className="eyebrow">Mission timeline</span>
        <div className="mission-legend">
          <LegendItem colorVar="--c-orchestrator" glyph="◆">reflection</LegendItem>
          <LegendItem colorVar="--c-takeover" glyph="⚑">takeover</LegendItem>
          <LegendItem colorVar="--c-promote" glyph="▣">decision</LegendItem>
        </div>
        <button type="button" className="text-button" aria-expanded={expanded}
          onClick={() => setExpanded(v => !v)}>{expanded ? 'Collapse ▴' : 'Expand ▾'}</button>
      </div>
      <div ref={wrapRef} className="mission-scroll">
        <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} role="img"
          aria-label="Swimlane map: orchestrator lane with notes, one lane per delegation with lifespan and cycle outcomes">
          <defs>
            <pattern id="fd-takeover-stripes" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
              <rect width="7" height="7" fill="transparent" />
              <line x1="0" y1="0" x2="0" y2="7" stroke="var(--c-takeover)" strokeWidth="2.5" opacity="0.55" />
            </pattern>
          </defs>
          {/* time grid */}
          {model.ticks.map(t => (
            <g key={t}>
              <line className="grid" x1={model.x(t)} x2={model.x(t)} y1={M.t} y2={height - M.b} />
              <text className="axis-label" x={model.x(t)} y={height - 6} textAnchor="middle">{clockLabel(t)}</text>
            </g>
          ))}
          {model.lanes.map((lane, i) => {
            const y = laneY(i);
            const dimmed = activeDelegationId != null && lane.kind === 'delegation' && lane.id !== activeDelegationId;
            const active = lane.kind === 'delegation' && lane.id === activeDelegationId;
            if (lane.kind === 'orch') {
              return (
                <g key={lane.id}>
                  <text className="lane-label" x={M.l - 10} y={y + 4} textAnchor="end" style={{ fill: `var(${lane.colorVar})` }}>{lane.label}</text>
                  <line x1={model.x(model.t0)} x2={model.x(model.t1)} y1={y} y2={y} style={{ stroke: `var(${lane.colorVar})` }} strokeWidth={2} opacity={0.5} />
                </g>
              );
            }
            if (lane.kind === 'playoff') {
              const es = model.playoffExps;
              const a = Math.min(...es.map(e => parseTime(e.created_at) ?? model.t1));
              const b = model.t1;
              return (
                <g key={lane.id} className="lane-group">
                  <text className="lane-label" x={M.l - 10} y={y + 4} textAnchor="end" style={{ fill: `var(${lane.colorVar})` }}>{lane.label}</text>
                  <rect className="lifespan" x={model.x(a)} y={y - 7} width={Math.max(8, model.x(b) - model.x(a))} height={14} rx={7}
                    style={{ fill: `var(${lane.colorVar})`, stroke: `var(${lane.colorVar})` }} role="link" tabIndex={0}
                    aria-label="Consolidation playoff — open journey finale"
                    onMouseMove={ev => show(ev, <><div className="tip-title">Playoff consolidation</div><div className="tip-meta mono">{es.length} replay experiments</div><div className="tip-hint">click → journey finale</div></>)}
                    onMouseLeave={hide} onClick={() => scrollTo('finale')}
                    onKeyDown={ev => ev.key === 'Enter' && scrollTo('finale')} />
                  {es.filter(e => !e.is_seed && !e.is_baseline).map(e => {
                    const t = parseTime(e.created_at); if (t == null) return null;
                    const o = outcomeOf(e);
                    return <circle key={e.experiment_id} className={`cycle-dot ${classNameFor(e.experiment_id)}`} cx={model.x(t)} cy={y} r={3.6} style={{ fill: `var(${OUTCOME_VAR[o]})` }}
                      onMouseEnter={ev => showExperiment(ev, e)} onMouseMove={moveExperiment} onMouseLeave={clearExperiment} />;
                  })}
                </g>
              );
            }
            const d = delegations.find(x => x.delegation_id === lane.id)!;
            const a = parseTime(d.spawned_at), b = parseTime(d.ended_at);
            if (a == null) return null;
            const bx0 = model.x(a), bx1 = model.x(b ?? model.t1);
            const es = experiments.filter(e => e.delegation_id === lane.id && !e.is_seed && !e.is_baseline);
            return (
              <g key={lane.id} className={`lane-group ${dimmed ? 'dimmed' : ''} ${active ? 'active' : ''}`}>
                <text className="lane-label" x={M.l - 10} y={y + 4} textAnchor="end" style={{ fill: `var(${lane.colorVar})` }}>{lane.label}</text>
                {/* spawn arrow from orchestrator lane */}
                <path className="spawn-arc" d={`M ${bx0} ${orchY + 3} C ${bx0} ${(orchY + y) / 2}, ${bx0} ${(orchY + y) / 2}, ${bx0} ${y - 9}`} style={{ stroke: `var(${lane.colorVar})` }} />
                <path d={`M ${bx0 - 3} ${y - 12} L ${bx0} ${y - 8} L ${bx0 + 3} ${y - 12}`} fill="none" style={{ stroke: `var(${lane.colorVar})` }} strokeWidth={1.2} opacity={0.6} />
                <rect className="lifespan" x={bx0} y={y - 7} width={Math.max(8, bx1 - bx0)} height={14} rx={7}
                  style={{ fill: `var(${lane.colorVar})`, stroke: `var(${lane.colorVar})` }} role="link" tabIndex={0}
                  aria-label={`${d.delegation_id} lifespan — open journey chapter`}
                  onMouseMove={ev => show(ev, delegationTip(d))} onMouseLeave={hide}
                  onClick={() => scrollTo(d.delegation_id)}
                  onKeyDown={ev => ev.key === 'Enter' && scrollTo(d.delegation_id)} />
                {d.taken_over && <rect x={bx0} y={y - 7} width={Math.max(8, bx1 - bx0)} height={14} rx={7} fill="url(#fd-takeover-stripes)" pointerEvents="none" />}
                {d.distress.active.length > 0 && <text className="distress-mark" x={bx1 + 6} y={y + 4}>✕</text>}
                {/* cycle ticks */}
                {es.map(e => {
                  const t = parseTime(e.created_at); if (t == null) return null;
                  const o = outcomeOf(e);
                  return (
                    <circle key={e.experiment_id} className={`cycle-dot ${classNameFor(e.experiment_id)}`} cx={Math.min(Math.max(model.x(t), bx0 + 4), bx1 - 4)} cy={y} r={3.6}
                      style={{ fill: `var(${OUTCOME_VAR[o]})` }} role="link" tabIndex={0}
                      aria-label={`${e.name}, ${OUTCOME_LABEL[o]} — open journey card`}
                      onMouseEnter={ev => showExperiment(ev, e)} onMouseMove={moveExperiment}
                      onMouseLeave={clearExperiment} onClick={() => { clearExperiment(); scrollTo(experimentHash(e)); }}
                      onKeyDown={ev => ev.key === 'Enter' && scrollTo(experimentHash(e))} />
                  );
                })}
              </g>
            );
          })}
          {/* takeover arcs: from the dead delegation's end back up to the orchestrator note */}
          {notes.filter(n => n.kind === 'takeover' && n.delegation_id && laneIndex.has(n.delegation_id)).map((n, i) => {
            const d = delegations.find(x => x.delegation_id === n.delegation_id);
            const endT = parseTime(d?.ended_at), noteT = parseTime(n.at);
            if (endT == null || noteT == null) return null;
            const y = laneY(laneIndex.get(n.delegation_id!)!);
            return <path key={i} className="takeover-arc" d={`M ${model.x(endT)} ${y - 8} C ${model.x(endT)} ${orchY + 26}, ${model.x(noteT)} ${orchY + 22}, ${model.x(noteT)} ${orchY + 7}`} />;
          })}
          {/* orchestrator notes + connector arcs */}
          {notes.map((n, i) => {
            const t = parseTime(n.at); if (t == null) return null;
            const glyph = NOTE_GLYPH[n.kind] ?? '◆';
            const colorVar = NOTE_VAR[n.kind] ?? '--c-orchestrator';
            const targetLane = n.delegation_id != null && laneIndex.has(n.delegation_id) ? laneIndex.get(n.delegation_id)! : null;
            const targetId = n.kind === 'decision' ? 'finale' : n.delegation_id ?? 'finale';
            return (
              <g key={`${n.at}-${i}`}>
                {targetLane != null && n.kind !== 'takeover' && (
                  <path className="note-arc" d={`M ${model.x(t)} ${orchY + 6} C ${model.x(t)} ${(orchY + laneY(targetLane)) / 2}, ${model.x(t)} ${(orchY + laneY(targetLane)) / 2}, ${model.x(t)} ${laneY(targetLane) - 10}`} />
                )}
                <text className="note-glyph" x={model.x(t)} y={orchY + 5} textAnchor="middle" tabIndex={0} role="link"
                  style={{ fill: `var(${colorVar})`, fontSize: n.kind === 'takeover' ? 15 : 13 }}
                  aria-label={`Orchestrator ${n.kind}${n.delegation_id ? ` about ${n.delegation_id}` : ''} — open in journey`}
                  onMouseMove={ev => show(ev, <><div className="tip-title" style={{ color: `var(${colorVar})` }}>{glyph} Orchestrator {n.kind}{n.delegation_id ? ` · re ${n.delegation_id}` : ''}</div><div className="tip-meta mono">{localTime(n.at)}</div><p className="tip-body">{n.text.length > 260 ? `${n.text.slice(0, 260)}…` : n.text}</p><div className="tip-hint">click → journey</div></>)}
                  onMouseLeave={hide} onClick={() => scrollTo(targetId)}
                  onKeyDown={ev => ev.key === 'Enter' && scrollTo(targetId)}>{glyph}</text>
              </g>
            );
          })}
        </svg>
      </div>
      {tooltip}
    </section>
  );
}
