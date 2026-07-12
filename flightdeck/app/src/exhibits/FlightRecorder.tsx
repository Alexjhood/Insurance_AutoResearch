import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { scaleLinear } from 'd3-scale';
import { select } from 'd3-selection';
import { zoom, zoomIdentity, type ZoomTransform, type D3ZoomEvent } from 'd3-zoom';
import { Drawer } from '../components/Drawer';
import type { DelegationTelemetry, ModelCallEvent, ToolCallEvent, WorkflowEvent } from '../lib/types';
import {
  SegmentToggle, useExhibitTooltip, useMeasure, useThemeVersion, cssVar,
  parseTime, clockLabel, localTime, durationLabel, humanTokens,
} from './shared';
import './exhibits.css';

export interface FlightRecorderProps { telemetry: DelegationTelemetry; selectedExperimentName?: string | null }

type RecorderEvent = {
  kind: 'model' | 'tool' | 'workflow';
  row: number; t0: number; t1: number;
  label: string; failed: boolean;
  heightFrac: number;              // model calls: output-token encoding; others 1
  source: ModelCallEvent | ToolCallEvent | WorkflowEvent;
};

const GUTTER = 118, ROW_H = 26, MODEL_ROW_H = 46, AXIS_H = 22, MINIMAP_H = 30, MINIMAP_GAP = 10;

interface RecorderModel {
  rows: { id: string; label: string; colorVar: string; h: number }[];
  rowTops: number[]; events: RecorderEvent[]; t0: number; t1: number; plotH: number;
  checkpoints: { name: string; t: number; cumulative: number }[];
}
const TOOL_VARS = ['--c-d1', '--c-d2', '--c-d3', '--c-d4', '--c-d5', '--c-d6', '--c-d7', '--c-d8'];

export function FlightRecorder({ telemetry, selectedExperimentName }: FlightRecorderProps) {
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const [selected, setSelected] = useState<RecorderEvent | null>(null);

  const model = useMemo(() => {
    const toolNames = [...new Set(telemetry.tool_calls.map(c => c.name))].sort(
      (a, b) => telemetry.tool_calls.filter(c => c.name === b).length - telemetry.tool_calls.filter(c => c.name === a).length);
    const rows: { id: string; label: string; colorVar: string; h: number }[] = [
      { id: '__model', label: 'model calls', colorVar: '--tk-reasoning', h: MODEL_ROW_H },
      ...toolNames.map((t, i) => ({ id: `tool:${t}`, label: t, colorVar: TOOL_VARS[i % TOOL_VARS.length], h: ROW_H })),
      { id: '__workflow', label: 'workflow', colorVar: '--accent', h: ROW_H },
    ];
    const rowIndex = new Map(rows.map((r, i) => [r.id, i]));
    const maxOut = Math.max(...telemetry.model_calls.map(c => c.output + c.reasoning), 1);
    const events: RecorderEvent[] = [];
    for (const c of telemetry.model_calls) {
      const t = parseTime(c.at); if (t == null) continue;
      events.push({ kind: 'model', row: rowIndex.get('__model')!, t0: t, t1: t, label: c.model, failed: false,
        heightFrac: Math.max(0.12, Math.sqrt((c.output + c.reasoning) / maxOut)), source: c });
    }
    for (const c of telemetry.tool_calls) {
      const a = parseTime(c.started_at); if (a == null) continue;
      const b = parseTime(c.completed_at) ?? (c.duration_ms != null ? a + c.duration_ms : a);
      events.push({ kind: 'tool', row: rowIndex.get(`tool:${c.name}`)!, t0: a, t1: b, label: c.name,
        failed: c.success === false || (c.status != null && /fail|error/i.test(c.status)), heightFrac: 1, source: c });
    }
    for (const c of telemetry.workflow_events) {
      const a = parseTime(c.started_at); if (a == null) continue;
      const b = parseTime(c.completed_at) ?? (c.duration_ms != null ? a + c.duration_ms : a);
      events.push({ kind: 'workflow', row: rowIndex.get('__workflow')!, t0: a, t1: b, label: c.command,
        failed: c.error_type != null || (c.status != null && /fail|error/i.test(c.status)), heightFrac: 1, source: c });
    }
    const times = events.flatMap(e => [e.t0, e.t1]);
    for (const cp of telemetry.checkpoints) { const t = parseTime(cp.completed_at); if (t != null) times.push(t); }
    const t0 = Math.min(...times), t1 = Math.max(...times);
    const rowTops: number[] = []; let acc = AXIS_H;
    for (const r of rows) { rowTops.push(acc); acc += r.h; }
    const checkpoints = telemetry.checkpoints
      .map(cp => ({ name: cp.experiment_name, t: parseTime(cp.completed_at), cumulative: cp.total_cumulative }))
      .filter((cp): cp is { name: string; t: number; cumulative: number } => cp.t != null);
    return { rows, rowTops, events, t0, t1, plotH: acc, checkpoints };
  }, [telemetry]);

  const failures = model.events.filter(e => e.failed).length;

  return (
    <section className="exhibit panel flight-recorder">
      <header className="section-title">
        <div><span className="eyebrow">Flight recorder</span><h2>Telemetry timeline · {telemetry.delegation_id}</h2></div>
        <div className="exhibit-controls">
          <span className="exhibit-note">{telemetry.tool_calls.length} tool calls · {telemetry.model_calls.length} model calls{failures ? <span style={{ color: 'var(--c-distress)' }}> · {failures} failed</span> : ''}</span>
          <SegmentToggle label="Flight recorder view" value={view} onChange={(v: 'chart' | 'table') => setView(v)}
            options={[{ value: 'chart', label: 'Timeline' }, { value: 'table', label: 'Table' }]} />
        </div>
      </header>
      {view === 'chart'
        ? <RecorderCanvas model={model} selectedExperimentName={selectedExperimentName ?? null} onSelect={setSelected} />
        : <RecorderTable telemetry={telemetry} />}
      {view === 'chart' && (
        <div className="exhibit-legend">
          {model.rows.filter(r => r.id.startsWith('tool:')).map(r => (
            <span className="legend-item" key={r.id}><i style={{ color: `var(${r.colorVar})` }}>■</i>{r.label}</span>
          ))}
          <span className="legend-item"><i style={{ color: 'var(--tk-reasoning)' }}>■</i>model call (height = output tokens)</span>
          <span className="legend-item"><i style={{ color: 'var(--accent)' }}>■</i>workflow command</span>
          <span className="legend-item"><i style={{ color: 'var(--c-distress)' }}>▢</i>failure</span>
          <span className="exhibit-note">wheel = zoom · drag = pan · shift-drag = brush zoom · click = inspect</span>
        </div>
      )}
      <EventDrawer event={selected} workflow={telemetry.workflow_events} onClose={() => setSelected(null)} />
    </section>
  );
}

/* ------------------------------------------------------------------ canvas */
function RecorderCanvas({ model, selectedExperimentName, onSelect }:
  { model: RecorderModel; selectedExperimentName: string | null; onSelect(e: RecorderEvent): void }) {
  const [wrapRef, width] = useMeasure<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const miniRef = useRef<HTMLCanvasElement>(null);
  const transformRef = useRef<ZoomTransform>(zoomIdentity);
  const hitsRef = useRef<{ x0: number; x1: number; y0: number; y1: number; ev: RecorderEvent }[]>([]);
  const brushRef = useRef<{ x0: number; x1: number } | null>(null);
  const { tooltip, show, hide } = useExhibitTooltip();
  const themeVersion = useThemeVersion();
  const w = Math.max(width, 480);
  const h = model.plotH + 6;

  const xBase = useMemo(() => scaleLinear([model.t0, model.t1], [GUTTER, w - 12]), [model, w]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d')!; ctx.scale(dpr, dpr);
    const xz = transformRef.current.rescaleX(xBase);
    const [d0, d1] = xz.domain() as [number, number];
    const colors = {
      grid: cssVar('--grid-line'), text: cssVar('--text-faint'), label: cssVar('--text-secondary'),
      inset: cssVar('--bg-inset'), raised: cssVar('--bg-raised-2'), distress: cssVar('--c-distress'),
      takeover: cssVar('--c-takeover'), border: cssVar('--border-subtle'),
      rows: model.rows.map(r => cssVar(r.colorVar)),
    };
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = colors.inset; ctx.fillRect(0, 0, w, h);
    ctx.font = '10.5px "JetBrains Mono", ui-monospace, monospace';

    // selected-experiment checkpoint segment shading
    if (selectedExperimentName) {
      const idx = model.checkpoints.findIndex(cp => cp.name === selectedExperimentName);
      if (idx >= 0) {
        const segStart = idx === 0 ? model.t0 : model.checkpoints[idx - 1].t;
        const segEnd = model.checkpoints[idx].t;
        ctx.fillStyle = colors.takeover; ctx.globalAlpha = 0.07;
        ctx.fillRect(xz(segStart), AXIS_H, xz(segEnd) - xz(segStart), h - AXIS_H);
        ctx.globalAlpha = 1;
      }
    }
    // time axis
    const span = d1 - d0;
    const stepMs = niceTimeStep(span / Math.max(4, Math.floor((w - GUTTER) / 130)));
    ctx.strokeStyle = colors.grid; ctx.fillStyle = colors.text; ctx.textAlign = 'center';
    for (let t = Math.ceil(d0 / stepMs) * stepMs; t <= d1; t += stepMs) {
      const px = xz(t);
      if (px < GUTTER) continue;
      ctx.beginPath(); ctx.moveTo(px, AXIS_H - 4); ctx.lineTo(px, h); ctx.stroke();
      ctx.fillText(span < 90_000 ? `${clockLabel(t)}:${String(new Date(t).getUTCSeconds()).padStart(2, '0')}` : clockLabel(t), px, 12);
    }
    // row separators + labels
    ctx.textAlign = 'left';
    model.rows.forEach((r, i) => {
      const top = model.rowTops[i];
      ctx.strokeStyle = colors.grid; ctx.beginPath(); ctx.moveTo(0, top); ctx.lineTo(w, top); ctx.stroke();
      ctx.fillStyle = colors.rows[i]; ctx.fillText(r.label.length > 14 ? `${r.label.slice(0, 13)}…` : r.label, 8, top + r.h / 2 + 4);
    });
    // events
    const hits: typeof hitsRef.current = [];
    for (const e of model.events) {
      if (e.t1 < d0 || e.t0 > d1) continue;
      const row = model.rows[e.row]; const top = model.rowTops[e.row];
      const x0 = Math.max(xz(e.t0), GUTTER), x1 = Math.min(xz(e.t1), w);
      const bw = Math.max(x1 - x0, 2);
      const bh = Math.max(4, (row.h - 10) * e.heightFrac);
      const y0 = top + (row.h - bh) / 2;
      ctx.fillStyle = colors.rows[e.row];
      ctx.globalAlpha = e.kind === 'model' ? 0.9 : 0.75;
      ctx.fillRect(x0, y0, bw, bh);
      ctx.globalAlpha = 1;
      if (e.failed) { ctx.strokeStyle = colors.distress; ctx.lineWidth = 1.5; ctx.strokeRect(x0 - 1, y0 - 1, bw + 2, bh + 2); ctx.lineWidth = 1; }
      hits.push({ x0: x0 - 2, x1: x0 + bw + 2, y0: top, y1: top + row.h, ev: e });
    }
    hitsRef.current = hits;
    // checkpoint markers (labels staggered on two lines to avoid collisions)
    model.checkpoints.forEach((cp, i) => {
      const px = xz(cp.t);
      if (px < GUTTER || px > w) return;
      ctx.strokeStyle = colors.takeover; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(px, AXIS_H); ctx.lineTo(px, h); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = colors.takeover; ctx.textAlign = 'right';
      ctx.fillText(cp.name.length > 26 ? `${cp.name.slice(0, 25)}…` : cp.name, px - 4, AXIS_H + 10 + (i % 2) * 11);
      ctx.textAlign = 'left';
    });
    // active brush overlay
    if (brushRef.current) {
      const { x0, x1 } = brushRef.current;
      ctx.fillStyle = colors.takeover; ctx.globalAlpha = 0.12;
      ctx.fillRect(Math.min(x0, x1), AXIS_H, Math.abs(x1 - x0), h - AXIS_H);
      ctx.globalAlpha = 1;
    }
    drawMinimap();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, w, h, xBase, selectedExperimentName, themeVersion]);

  const drawMinimap = useCallback(() => {
    const canvas = miniRef.current; if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = MINIMAP_H * dpr;
    const ctx = canvas.getContext('2d')!; ctx.scale(dpr, dpr);
    ctx.fillStyle = cssVar('--bg-inset'); ctx.fillRect(0, 0, w, MINIMAP_H);
    const mx = scaleLinear([model.t0, model.t1], [GUTTER, w - 12]);
    for (const e of model.events) {
      ctx.fillStyle = cssVar(model.rows[e.row].colorVar); ctx.globalAlpha = 0.5;
      ctx.fillRect(mx(e.t0), 4 + (e.row / model.rows.length) * (MINIMAP_H - 8), Math.max(mx(e.t1) - mx(e.t0), 1), Math.max((MINIMAP_H - 8) / model.rows.length, 2));
    }
    ctx.globalAlpha = 1;
    const xz = transformRef.current.rescaleX(xBase);
    const [d0, d1] = xz.domain() as [number, number];
    ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 1.5;
    ctx.strokeRect(mx(Math.max(d0, model.t0)), 1, Math.max(mx(Math.min(d1, model.t1)) - mx(Math.max(d0, model.t0)), 4), MINIMAP_H - 2);
  }, [model, w, xBase]);

  // zoom behavior
  const zoomBehaviorRef = useRef<ReturnType<typeof zoom<HTMLCanvasElement, unknown>> | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const zb = zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([1, 2000])
      .translateExtent([[GUTTER, 0], [w - 12, h]])
      .extent([[GUTTER, 0], [w - 12, h]])
      .filter(ev => !(ev as MouseEvent).shiftKey && (ev as MouseEvent).button === 0)
      .on('zoom', (ev: D3ZoomEvent<HTMLCanvasElement, unknown>) => { transformRef.current = ev.transform; draw(); });
    zoomBehaviorRef.current = zb;
    const sel = select(canvas);
    sel.call(zb);
    sel.call(zb.transform, transformRef.current);
    return () => { sel.on('.zoom', null); };
  }, [w, h, draw]);

  useEffect(() => { draw(); }, [draw]);

  // hover, click, shift-brush
  const onMouseMove = (ev: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    if (brushRef.current) { brushRef.current.x1 = px; draw(); return; }
    const hit = hitAt(hitsRef.current, px, py);
    ev.currentTarget.style.cursor = hit ? 'pointer' : 'grab';
    if (hit) show(ev, eventTip(hit.ev)); else hide();
  };
  const onMouseDown = (ev: React.MouseEvent<HTMLCanvasElement>) => {
    if (!ev.shiftKey) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    brushRef.current = { x0: ev.clientX - rect.left, x1: ev.clientX - rect.left };
  };
  const onMouseUp = () => {
    const canvas = canvasRef.current!;
    if (!brushRef.current) return;
    const { x0, x1 } = brushRef.current; brushRef.current = null;
    const [a, b] = [Math.min(x0, x1), Math.max(x0, x1)];
    if (b - a > 8 && zoomBehaviorRef.current) {
      const xz = transformRef.current.rescaleX(xBase);
      const [ta, tb] = [xz.invert(a), xz.invert(b)];
      const k = Math.min(2000, (model.t1 - model.t0) / Math.max(tb - ta, 1));
      select(canvas).call(zoomBehaviorRef.current.transform, zoomIdentity.translate(GUTTER, 0).scale(k).translate(-xBase(ta), 0));
      return;
    }
    draw();
  };
  // d3-zoom suppresses mouseup propagation (capture-phase window handler), but leaves no-drag clicks alone.
  const onClick = (ev: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    const hit = hitAt(hitsRef.current, px, py);
    if (hit) { hide(); onSelect(hit.ev); }
  };
  const onMiniDown = (ev: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const mx = scaleLinear([GUTTER, w - 12], [model.t0, model.t1]);
    const t = mx(ev.clientX - rect.left);
    const canvas = canvasRef.current;
    if (canvas && zoomBehaviorRef.current) {
      select(canvas).call(zoomBehaviorRef.current.translateTo, xBase(t), 0, [(GUTTER + w - 12) / 2, 0]);
    }
  };

  return (
    <div ref={wrapRef} className="recorder-wrap">
      <canvas ref={canvasRef} style={{ width: w, height: h }} role="img"
        aria-label="Zoomable Gantt of model calls, tool calls and workflow commands over the delegation's lifetime"
        onMouseMove={onMouseMove} onMouseLeave={() => { hide(); brushRef.current = null; }}
        onMouseDown={onMouseDown} onMouseUp={onMouseUp} onClick={onClick} />
      <canvas ref={miniRef} className="recorder-minimap" style={{ width: w, height: MINIMAP_H, marginTop: MINIMAP_GAP }}
        aria-label="Minimap — click to pan the timeline" onMouseDown={onMiniDown} />
      {tooltip}
    </div>
  );
}

type HitBox = { x0: number; x1: number; y0: number; y1: number; ev: RecorderEvent };
function hitAt(hits: HitBox[], px: number, py: number): HitBox | undefined {
  for (let i = hits.length - 1; i >= 0; i--) {
    const hb = hits[i];
    if (px >= hb.x0 && px <= hb.x1 && py >= hb.y0 && py <= hb.y1) return hb;
  }
  return undefined;
}

function niceTimeStep(target: number): number {
  const steps = [1000, 5000, 15_000, 30_000, 60_000, 120_000, 300_000, 600_000, 1_200_000, 1_800_000, 3_600_000];
  return steps.find(s => s >= target) ?? 7_200_000;
}

function eventTip(e: RecorderEvent) {
  const dur = e.t1 > e.t0 ? durationLabel(e.t1 - e.t0) : null;
  const meta: string[] = [];
  if (e.kind === 'model') {
    const c = e.source as ModelCallEvent;
    meta.push(`in ${humanTokens(c.input)} (${c.input ? Math.round(100 * c.cached_input / c.input) : 0}% cached)`, `out ${humanTokens(c.output)}`, `reasoning ${humanTokens(c.reasoning)}`);
  } else if (e.kind === 'tool') {
    const c = e.source as ToolCallEvent;
    if (dur) meta.push(dur);
    if (c.output_bytes != null) meta.push(`${humanTokens(c.output_bytes)}B out`);
    if (c.status) meta.push(c.status);
  } else {
    const c = e.source as WorkflowEvent;
    if (dur) meta.push(dur);
    if (c.status) meta.push(c.status);
  }
  return (
    <>
      <div className="tip-title">{e.kind === 'workflow' ? `autoresearch ${e.label}` : e.label}</div>
      <div className="tip-meta mono">{localTime(new Date(e.t0).toISOString())} · {meta.join(' · ')}</div>
      {e.failed && <div className="tip-meta" style={{ color: 'var(--c-distress)' }}>✕ failed</div>}
      <div className="tip-hint">click → detail drawer</div>
    </>
  );
}

/* ------------------------------------------------------------------ drawer */
function EventDrawer({ event, workflow, onClose }: { event: RecorderEvent | null; workflow: WorkflowEvent[]; onClose(): void }) {
  if (!event) return null;
  const rows: [string, React.ReactNode][] = [];
  rows.push(['Kind', event.kind], ['Started', localTime(new Date(event.t0).toISOString())]);
  if (event.t1 > event.t0) rows.push(['Duration', durationLabel(event.t1 - event.t0)]);
  if (event.kind === 'model') {
    const c = event.source as ModelCallEvent;
    rows.push(['Model', c.model], ['Input tokens', humanTokens(c.input)],
      ['Cached input', `${humanTokens(c.cached_input)} (${c.input ? Math.round(100 * c.cached_input / c.input) : 0}%)`],
      ['Output tokens', humanTokens(c.output)], ['Reasoning tokens', humanTokens(c.reasoning)]);
    const wf = workflow.find(x => x.id === c.workflow_event_id);
    if (wf) rows.push(['Workflow command', `autoresearch ${wf.command}`]);
  } else if (event.kind === 'tool') {
    const c = event.source as ToolCallEvent;
    rows.push(['Status', c.status ?? '—'], ['Input bytes', c.input_bytes ?? '—'], ['Output bytes', c.output_bytes ?? '—']);
    if (c.error_type) rows.push(['Error type', c.error_type]);
    if (c.detail) rows.push(['Detail', <code key="d" className="drawer-code">{c.detail.slice(0, 400)}</code>]);
    const wf = workflow.find(x => x.id === c.workflow_event_id);
    if (wf) rows.push(['Workflow command', `autoresearch ${wf.command}`]);
  } else {
    const c = event.source as WorkflowEvent;
    rows.push(['Command', `autoresearch ${c.command}`], ['Status', c.status ?? '—']);
    if (c.error_type) rows.push(['Error type', c.error_type]);
  }
  return (
    <Drawer title={event.kind === 'workflow' ? `autoresearch ${event.label}` : event.label} open onClose={onClose}>
      <dl className="drawer-detail">
        {rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{v}{event.failed && k === 'Status' ? ' ✕' : ''}</dd></div>)}
      </dl>
    </Drawer>
  );
}

/* ------------------------------------------------------------------ table */
function RecorderTable({ telemetry }: { telemetry: DelegationTelemetry }) {
  const parent = useRef<HTMLDivElement>(null);
  const rows = telemetry.tool_calls;
  const v = useVirtualizer({ count: rows.length, getScrollElement: () => parent.current, estimateSize: () => 36, overscan: 8 });
  return (
    <>
      <div className="recorder-head"><span>Tool</span><span>Started</span><span>Duration</span><span>Status</span></div>
      <div className="recorder-scroll" ref={parent}>
        <div style={{ height: v.getTotalSize(), position: 'relative' }}>
          {v.getVirtualItems().map(item => {
            const call = rows[item.index];
            return (
              <div className="recorder-row" key={item.key} style={{ transform: `translateY(${item.start}px)` }}>
                <span>{call.name}</span>
                <span>{call.started_at ? localTime(call.started_at) : '—'}</span>
                <span>{call.duration_ms == null ? '—' : durationLabel(call.duration_ms)}</span>
                <span className={call.success === false ? 'failed' : ''}>{call.success === false ? '✕ ' : ''}{call.status ?? 'unknown'}</span>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}
