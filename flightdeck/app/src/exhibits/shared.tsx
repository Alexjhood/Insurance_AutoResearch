import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { PropsWithChildren, ReactNode } from 'react';
import './exhibits.css';

/* ---------- time ---------- */
/** Registry timestamps come as "2026-07-11 16:53:14" (UTC, no zone) or ISO-8601. */
export function parseTime(value: string | null | undefined): number | null {
  if (!value) return null;
  const iso = value.includes('T') ? value : `${value.replace(' ', 'T')}Z`;
  const t = Date.parse(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  return Number.isNaN(t) ? null : t;
}
export function clockLabel(ms: number) {
  const d = new Date(ms);
  return `${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')}`;
}
export function localTime(value: string | null | undefined) {
  const t = parseTime(value);
  return t == null ? '—' : new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
export function durationLabel(ms: number | null | undefined) {
  if (ms == null) return '—';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

/* ---------- layout / environment hooks ---------- */
export function useMeasure<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    setWidth(node.clientWidth);
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) setWidth(entry.contentRect.width);
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() =>
    typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    const mq = matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

/** Bumps whenever data-theme changes, so canvas exhibits re-resolve CSS variables. */
export function useThemeVersion(): number {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const mo = new MutationObserver(() => setVersion(v => v + 1));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, []);
  return version;
}

/** Resolve a CSS custom property to its concrete color (for canvas drawing). */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ---------- semantic vocabulary (SPEC §4) ---------- */
export type Outcome = 'promote' | 'local_promote' | 'reject' | 'forfeited' | 'seed' | 'baseline' | 'playoff';
export function outcomeOf(e: { is_seed: boolean; is_baseline: boolean; delegation_id: string | null; comparison: { decision: string | null } | null }): Outcome {
  if (e.is_baseline) return 'baseline';
  if (e.is_seed) return 'seed';
  const d = e.comparison?.decision;
  if (d === 'promote' || d === 'local_promote' || d === 'reject') {
    return e.delegation_id == null ? 'playoff' : (d as Outcome);
  }
  return e.delegation_id == null ? 'playoff' : 'forfeited';
}
export const OUTCOME_GLYPH: Record<Outcome, string> = {
  promote: '▲', local_promote: '◭', reject: '○', forfeited: '✕', seed: '⇢', baseline: '◇', playoff: '◈',
};
export const OUTCOME_VAR: Record<Outcome, string> = {
  promote: '--c-promote', local_promote: '--c-localpromote', reject: '--c-reject',
  forfeited: '--c-distress', seed: '--c-baseline', baseline: '--c-baseline', playoff: '--c-orchestrator',
};
export const OUTCOME_LABEL: Record<Outcome, string> = {
  promote: 'promote', local_promote: 'local promote', reject: 'reject',
  forfeited: 'forfeited / undecided', seed: 'seed transfer', baseline: 'baseline', playoff: 'playoff replay',
};

/** Delegation identity color: d01 → --c-d1 … (SPEC §4, stable across all exhibits). */
export function delegationVar(delegationId: string | null): string {
  if (!delegationId) return '--c-orchestrator';
  const n = Number(delegationId.replace(/\D/g, ''));
  return n >= 1 && n <= 8 ? `--c-d${n}` : '--c-baseline';
}

/* ---------- tooltip ---------- */
interface TipState { x: number; y: number; body: ReactNode }
export function useExhibitTooltip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = useCallback((ev: { clientX: number; clientY: number }, body: ReactNode) => {
    const x = Math.min(ev.clientX + 14, window.innerWidth - 316);
    const y = Math.min(ev.clientY + 14, window.innerHeight - 140);
    setTip({ x, y, body });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const tooltip = tip ? (
    <div className="exhibit-tip" role="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.body}</div>
  ) : null;
  return { tooltip, show, hide };
}

/* ---------- exhibit chrome ---------- */
export function SegmentToggle<T extends string>({ options, value, onChange, label }:
  { options: { value: T; label: string }[]; value: T; onChange(v: T): void; label: string }) {
  return (
    <div className="seg-toggle" role="group" aria-label={label}>
      {options.map(o => (
        <button key={o.value} type="button" className={o.value === value ? 'on' : ''}
          aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export function ExhibitShell({ eyebrow, title, note, controls, legend, table, tableCaption, children, className }:
  PropsWithChildren<{
    eyebrow: string; title: string; note?: ReactNode; controls?: ReactNode;
    legend?: ReactNode; table: ReactNode; tableCaption: string; className?: string;
  }>) {
  const [view, setView] = useState<'chart' | 'table'>('chart');
  return (
    <section className={`exhibit panel ${className ?? ''}`}>
      <header className="section-title">
        <div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div>
        <div className="exhibit-controls">
          {note && <span className="exhibit-note">{note}</span>}
          {view === 'chart' && controls}
          <SegmentToggle label={`${title} view`} value={view} onChange={v => setView(v)}
            options={[{ value: 'chart', label: 'Chart' }, { value: 'table', label: 'Table' }]} />
        </div>
      </header>
      {view === 'chart' ? (
        <>{children}{legend && <div className="exhibit-legend">{legend}</div>}</>
      ) : (
        <div className="exhibit-table-wrap"><table className="exhibit-table"><caption className="sr-only">{tableCaption}</caption>{table}</table></div>
      )}
    </section>
  );
}

export function LegendItem({ colorVar, glyph, children }: PropsWithChildren<{ colorVar: string; glyph?: string }>) {
  return <span className="legend-item"><i style={{ color: `var(${colorVar})` }}>{glyph ?? '■'}</i>{children}</span>;
}

/* ---------- misc ---------- */
export function humanTokens(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}
