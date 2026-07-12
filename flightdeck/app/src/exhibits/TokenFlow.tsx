import { useMemo } from 'react';
import { scaleLinear } from 'd3-scale';
import { area, line, curveMonotoneX } from 'd3-shape';
import type { Delegation, Experiment, TelemetryByDelegation, TokenTotals } from '../lib/types';
import {
  ExhibitShell, LegendItem, useExhibitTooltip, useMeasure, useReducedMotion,
  delegationVar, humanTokens,
} from './shared';
import './exhibits.css';

export type TokenFlowProps =
  | { mode: 'campaign'; byDelegation: TelemetryByDelegation[] }
  | { mode: 'delegation'; delegation: Delegation; experiments: Experiment[] };

export function TokenFlow(props: TokenFlowProps) {
  return props.mode === 'campaign'
    ? <CampaignFlow byDelegation={props.byDelegation} />
    : <DelegationBurn delegation={props.delegation} experiments={props.experiments} />;
}

const CLASSES = [
  { key: 'cached', label: 'cached input', colorVar: '--tk-cached' },
  { key: 'uncached', label: 'uncached input', colorVar: '--tk-uncached' },
  { key: 'output', label: 'output', colorVar: '--tk-output' },
  { key: 'reasoning', label: 'reasoning', colorVar: '--tk-reasoning' },
] as const;
type ClassKey = typeof CLASSES[number]['key'];

function split(t: TokenTotals): Record<ClassKey, number> {
  return { cached: t.cached_input, uncached: Math.max(0, t.input - t.cached_input), output: t.output, reasoning: t.reasoning };
}
function grand(t: TokenTotals) { return t.input + t.output + t.reasoning; }

/* ================================================================ campaign */
function CampaignFlow({ byDelegation }: { byDelegation: TelemetryByDelegation[] }) {
  const [wrapRef, width] = useMeasure<HTMLDivElement>();
  const { tooltip, show, hide } = useExhibitTooltip();
  const reduced = useReducedMotion();
  const w = Math.max(width, 640);
  const H = 340;
  const M = { l: 10, r: 10, t: 26, b: 10 };
  const NODE_W = 12, GAP = 8;

  const model = useMemo(() => {
    const totals = byDelegation.map(d => ({ d, total: grand(d.tokens), parts: split(d.tokens) }));
    const sum = totals.reduce((n, x) => n + x.total, 0) || 1;
    const usable = H - M.t - M.b;
    const midH = usable - GAP * (totals.length - 1);
    const scale = midH / sum;
    const px = (v: number) => Math.max(v * scale, 1.5);

    // middle column: delegation nodes
    let y = M.t;
    const mids = totals.map(x => {
      const h = px(x.total);
      const node = { ...x, y, h };
      y += h + GAP;
      return node;
    });
    // left column: campaign node spans the summed heights (no gaps)
    const leftH = mids.reduce((n, m) => n + m.h, 0);
    const leftY = M.t + (usable - leftH) / 2;
    // right column: token-class nodes
    const classTotals = CLASSES.map(c => ({ ...c, total: totals.reduce((n, x) => n + x.parts[c.key], 0) }));
    const rightUsable = usable - GAP * (CLASSES.length - 1);
    const rightScale = rightUsable / sum;
    let ry = M.t;
    const rights = classTotals.map(c => {
      const h = Math.max(c.total * rightScale, 2);
      const node = { ...c, y: ry, h };
      ry += h + GAP;
      return node;
    });
    const xL = M.l, xM_ = w * 0.42, xR = w - M.r - NODE_W;
    return { mids, rights, leftY, leftH, sum, scale, rightScale, xL, xM: xM_, xR, px };
  }, [byDelegation, w]);

  const ribbon = (x0: number, y0: number, h0: number, x1: number, y1: number, h1: number) => {
    const c = (x0 + x1) / 2;
    return `M ${x0} ${y0} C ${c} ${y0}, ${c} ${y1}, ${x1} ${y1} L ${x1} ${y1 + h1} C ${c} ${y1 + h1}, ${c} ${y0 + h0}, ${x0} ${y0 + h0} Z`;
  };

  const totalsAll = byDelegation.reduce((acc, d) => ({
    input: acc.input + d.tokens.input, cached_input: acc.cached_input + d.tokens.cached_input,
    output: acc.output + d.tokens.output, reasoning: acc.reasoning + d.tokens.reasoning,
  }), { input: 0, cached_input: 0, output: 0, reasoning: 0 });
  const cachePct = totalsAll.input ? (100 * totalsAll.cached_input / totalsAll.input) : 0;
  const ratio = totalsAll.output ? Math.round(totalsAll.input / totalsAll.output) : 0;

  const delegationTip = (m: typeof model.mids[number]) => (
    <>
      <div className="tip-title" style={{ color: `var(${delegationVar(m.d.delegation_id)})` }}>{m.d.delegation_id}</div>
      <div className="tip-meta mono">total {humanTokens(m.total)} · {m.d.model_calls} model calls · {m.d.tool_calls} tool calls</div>
      <table className="tip-ledger"><tbody>
        {CLASSES.map(c => <tr key={c.key}><td><i style={{ color: `var(${c.colorVar})` }}>■</i> {c.label}</td><td className="num">{humanTokens(m.parts[c.key])}</td></tr>)}
        <tr><td>cache hit</td><td className="num">{m.d.cache_hit_rate == null ? '—' : `${(m.d.cache_hit_rate * 100).toFixed(1)}%`}</td></tr>
      </tbody></table>
    </>
  );

  const table = (
    <>
      <thead><tr><th>Delegation</th><th className="num">Total</th><th className="num">Cached input</th><th className="num">Uncached input</th><th className="num">Output</th><th className="num">Reasoning</th><th className="num">Cache hit</th></tr></thead>
      <tbody>
        {byDelegation.map(d => {
          const p = split(d.tokens);
          return <tr key={d.delegation_id}>
            <td><span style={{ color: `var(${delegationVar(d.delegation_id)})` }}>{d.delegation_id}</span></td>
            <td className="num">{grand(d.tokens).toLocaleString()}</td>
            <td className="num">{p.cached.toLocaleString()}</td><td className="num">{p.uncached.toLocaleString()}</td>
            <td className="num">{p.output.toLocaleString()}</td><td className="num">{p.reasoning.toLocaleString()}</td>
            <td className="num">{d.cache_hit_rate == null ? '—' : `${(d.cache_hit_rate * 100).toFixed(1)}%`}</td>
          </tr>;
        })}
      </tbody>
    </>
  );

  return (
    <ExhibitShell
      eyebrow="Token flow" title="Where the tokens went"
      note={<>{humanTokens(grand(totalsAll))} total · <b style={{ color: 'var(--tk-cached)' }}>{cachePct.toFixed(1)}% cache hits</b> · input ≈ {ratio}× output</>}
      legend={<>{CLASSES.map(c => <LegendItem key={c.key} colorVar={c.colorVar}>{c.label}</LegendItem>)}</>}
      table={table} tableCaption="Token totals by delegation and class"
    >
      <div className="chart-well" ref={wrapRef}>
        <svg className={reduced ? '' : 'animate-in'} width="100%" height={H} viewBox={`0 0 ${w} ${H}`} role="img"
          aria-label="Sankey-style token flow from the campaign through delegations into cached input, uncached input, output and reasoning tokens">
          <text className="col-label" x={model.xL} y={16}>campaign</text>
          <text className="col-label" x={model.xM} y={16}>delegations</text>
          <text className="col-label" x={model.xR + NODE_W} y={16} textAnchor="end">token class</text>
          {/* campaign node */}
          <rect x={model.xL} y={model.leftY} width={NODE_W} height={model.leftH} rx={3} fill="var(--border-strong)" />
          {/* campaign → delegation ribbons, split-shaded cached (calm) / rest (bright) */}
          {(() => {
            let srcY = model.leftY;
            return model.mids.map(m => {
              const cachedH = (m.parts.cached / m.total) * m.h;
              const el = (
                <g key={m.d.delegation_id} className="flow-ribbon"
                  onMouseMove={ev => show(ev, delegationTip(m))} onMouseLeave={hide}>
                  <path d={ribbon(model.xL + NODE_W, srcY, cachedH, model.xM, m.y, cachedH)}
                    style={{ fill: `var(${delegationVar(m.d.delegation_id)})` }} opacity={0.28} />
                  <path d={ribbon(model.xL + NODE_W, srcY + cachedH, m.h - cachedH, model.xM, m.y + cachedH, m.h - cachedH)}
                    style={{ fill: `var(${delegationVar(m.d.delegation_id)})` }} opacity={0.72} />
                </g>
              );
              srcY += m.h;
              return el;
            });
          })()}
          {/* delegation nodes */}
          {model.mids.map(m => (
            <g key={m.d.delegation_id} onMouseMove={ev => show(ev, delegationTip(m))} onMouseLeave={hide}>
              <rect x={model.xM} y={m.y} width={NODE_W} height={m.h} rx={3} style={{ fill: `var(${delegationVar(m.d.delegation_id)})` }} />
              <text className="node-label" x={model.xM + NODE_W + 6} y={m.y + Math.min(m.h / 2 + 4, m.h + 4)}
                style={{ fill: `var(${delegationVar(m.d.delegation_id)})` }}>{m.d.delegation_id} · {humanTokens(m.total)}</text>
            </g>
          ))}
          {/* delegation → class ribbons */}
          {(() => {
            const cursors = new Map(model.rights.map(r => [r.key, r.y]));
            return model.mids.flatMap(m => {
              let srcY = m.y;
              return CLASSES.map(c => {
                const v = m.parts[c.key];
                const h0 = (v / m.total) * m.h;
                const h1 = Math.max(v * model.rightScale, v > 0 ? 0.8 : 0);
                const destY = cursors.get(c.key)!;
                cursors.set(c.key, destY + h1);
                const el = v > 0 ? (
                  <path key={`${m.d.delegation_id}-${c.key}`} className="flow-ribbon"
                    d={ribbon(model.xM + NODE_W, srcY, h0, model.xR, destY, h1)}
                    style={{ fill: `var(${c.colorVar})` }} opacity={0.45}
                    onMouseMove={ev => show(ev, <><div className="tip-title">{m.d.delegation_id} → {c.label}</div><div className="tip-meta mono">{humanTokens(v)} tokens · {(100 * v / m.total).toFixed(v / m.total < 0.01 ? 2 : 1)}% of {m.d.delegation_id}</div></>)}
                    onMouseLeave={hide} />
                ) : null;
                srcY += h0;
                return el;
              });
            });
          })()}
          {/* class nodes (labels pushed apart so tiny nodes stay legible, kept inside the frame) */}
          {(() => {
            const ys = model.rights.map(r => r.y + Math.max(r.h / 2, 4) + 4);
            for (let i = 1; i < ys.length; i++) ys[i] = Math.max(ys[i], ys[i - 1] + 13);
            ys[ys.length - 1] = Math.min(ys[ys.length - 1], H - 8);
            for (let i = ys.length - 2; i >= 0; i--) ys[i] = Math.min(ys[i], ys[i + 1] - 13);
            return model.rights.map((r, i) => ({ r, labelY: ys[i] }));
          })().map(({ r, labelY }) => (
            <g key={r.key}
              onMouseMove={ev => show(ev, <><div className="tip-title">{r.label}</div><div className="tip-meta mono">{humanTokens(r.total)} tokens · {(100 * r.total / model.sum).toFixed(r.total / model.sum < 0.01 ? 2 : 1)}% of campaign</div></>)}
              onMouseLeave={hide}>
              <rect x={model.xR} y={r.y} width={NODE_W} height={r.h} rx={3} style={{ fill: `var(${r.colorVar})` }} />
              <text className="node-label" x={model.xR - 6} y={labelY} textAnchor="end"
                style={{ fill: `var(${r.colorVar})` }}>{r.label} · {humanTokens(r.total)}</text>
            </g>
          ))}
        </svg>
        {tooltip}
      </div>
    </ExhibitShell>
  );
}

/* ============================================================== delegation */
function DelegationBurn({ delegation, experiments }: { delegation: Delegation; experiments: Experiment[] }) {
  const [wrapRef, width] = useMeasure<HTMLDivElement>();
  const { tooltip, show, hide } = useExhibitTooltip();
  const reduced = useReducedMotion();
  const w = Math.max(width, 560);
  const H = 300;
  const M = { l: 62, r: 52, t: 16, b: 54 };

  const model = useMemo(() => {
    const steps = experiments.filter(e => e.usage != null).sort((a, b) => a.seq - b.seq)
      .map(e => ({ e, parts: split(e.usage!.tokens), cacheRate: e.usage!.cache_hit_rate }));
    // cumulative stacks: cached, uncached, output, reasoning
    let acc: Record<ClassKey, number> = { cached: 0, uncached: 0, output: 0, reasoning: 0 };
    const cum = steps.map(s => {
      acc = { cached: acc.cached + s.parts.cached, uncached: acc.uncached + s.parts.uncached, output: acc.output + s.parts.output, reasoning: acc.reasoning + s.parts.reasoning };
      return { ...s, cum: { ...acc }, total: acc.cached + acc.uncached + acc.output + acc.reasoning };
    });
    const maxTotal = Math.max(...cum.map(c => c.total), 1);
    const x = scaleLinear([0, Math.max(cum.length - 1, 1)], [M.l, w - M.r]);
    const y = scaleLinear([0, maxTotal * 1.05], [H - M.b, M.t]);
    const yPct = scaleLinear([0, 1], [H - M.b, M.t]);
    // stacked layers bottom-up in CLASSES order
    const layers = CLASSES.map((c, ci) => {
      const a = area<typeof cum[number]>()
        .x((_, i) => x(i))
        .y0(d => y(CLASSES.slice(0, ci).reduce((n, cc) => n + d.cum[cc.key], 0)))
        .y1(d => y(CLASSES.slice(0, ci + 1).reduce((n, cc) => n + d.cum[cc.key], 0)))
        .curve(curveMonotoneX);
      return { ...c, path: a(cum) ?? '' };
    });
    const cacheLine = line<typeof cum[number]>()
      .defined(d => d.cacheRate != null)
      .x((_, i) => x(i)).y(d => yPct(d.cacheRate ?? 0)).curve(curveMonotoneX)(cum) ?? '';
    return { cum, x, y, yPct, layers, cacheLine, maxTotal };
  }, [experiments, w]);

  const table = (
    <>
      <thead><tr><th>Experiment step</th><th className="num">Cached</th><th className="num">Uncached</th><th className="num">Output</th><th className="num">Reasoning</th><th className="num">Cache hit</th><th className="num">Cumulative</th></tr></thead>
      <tbody>
        {model.cum.map(s => (
          <tr key={s.e.experiment_id}>
            <td className="mono">{s.e.name}</td>
            <td className="num">{s.parts.cached.toLocaleString()}</td><td className="num">{s.parts.uncached.toLocaleString()}</td>
            <td className="num">{s.parts.output.toLocaleString()}</td><td className="num">{s.parts.reasoning.toLocaleString()}</td>
            <td className="num">{s.cacheRate == null ? '—' : `${(s.cacheRate * 100).toFixed(1)}%`}</td>
            <td className="num">{s.total.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </>
  );

  const totalTok = grand(delegation.cost.tokens);
  return (
    <ExhibitShell
      eyebrow="Token burn" title={`Cumulative burn · ${delegation.delegation_id}`}
      note={<>{humanTokens(totalTok)} total · {delegation.cost.cache_hit_rate == null ? '—' : `${(delegation.cost.cache_hit_rate * 100).toFixed(1)}%`} cache hit</>}
      legend={<>
        {CLASSES.map(c => <LegendItem key={c.key} colorVar={c.colorVar}>{c.label}</LegendItem>)}
        <LegendItem colorVar="--c-takeover" glyph="—">cache-hit %</LegendItem>
      </>}
      table={table} tableCaption="Per-experiment token usage checkpoints"
    >
      <div className="chart-well" ref={wrapRef}>
        {model.cum.length === 0 ? <p className="exhibit-note" style={{ padding: 'var(--sp-4)' }}>No usage checkpoints recorded for this delegation.</p> : (
          <svg className={reduced ? '' : 'animate-in'} width="100%" height={H} viewBox={`0 0 ${w} ${H}`} role="img"
            aria-label="Cumulative token area chart across experiment checkpoints with cache-hit rate overlay">
            {model.y.ticks(4).map(t => (
              <g key={t}>
                <line className="grid" x1={M.l} x2={w - M.r} y1={model.y(t)} y2={model.y(t)} />
                <text className="axis-label" x={M.l - 8} y={model.y(t) + 4} textAnchor="end">{humanTokens(t)}</text>
              </g>
            ))}
            {[0, 0.5, 1].map(p => (
              <text key={p} className="axis-label" x={w - M.r + 8} y={model.yPct(p) + 4} style={{ fill: 'var(--c-takeover)' }}>{p * 100}%</text>
            ))}
            {model.layers.map(l => <path key={l.key} d={l.path} style={{ fill: `var(${l.colorVar})` }} opacity={0.85} />)}
            <path d={model.cacheLine} fill="none" style={{ stroke: 'var(--c-takeover)' }} strokeWidth={1.8} strokeDasharray="5 4" />
            {/* checkpoint markers */}
            {model.cum.map((s, i) => (
              <g key={s.e.experiment_id} className="burn-checkpoint"
                onMouseMove={ev => show(ev, <>
                  <div className="tip-title">{s.e.name}</div>
                  <div className="tip-meta mono">step {i + 1}{s.e.cycle != null ? ` · cycle ${s.e.cycle}` : ''}</div>
                  <table className="tip-ledger"><tbody>
                    {CLASSES.map(c => <tr key={c.key}><td><i style={{ color: `var(${c.colorVar})` }}>■</i> {c.label}</td><td className="num">+{humanTokens(s.parts[c.key])}</td></tr>)}
                    <tr><td>cache hit</td><td className="num">{s.cacheRate == null ? '—' : `${(s.cacheRate * 100).toFixed(1)}%`}</td></tr>
                    <tr><td>cumulative</td><td className="num">{humanTokens(s.total)}</td></tr>
                  </tbody></table>
                </>)} onMouseLeave={hide}>
                <line x1={model.x(i)} x2={model.x(i)} y1={M.t} y2={H - M.b} className="checkpoint-rule" />
                <circle cx={model.x(i)} cy={model.y(s.total)} r={3.5} style={{ fill: 'var(--text-primary)' }} />
                <text className="axis-label checkpoint-label" x={model.x(i)} y={H - M.b + 14} textAnchor="end"
                  transform={`rotate(-24 ${model.x(i)} ${H - M.b + 14})`}>{s.e.name.length > 24 ? `${s.e.name.slice(0, 23)}…` : s.e.name}</text>
              </g>
            ))}
          </svg>
        )}
        {tooltip}
      </div>
    </ExhibitShell>
  );
}
