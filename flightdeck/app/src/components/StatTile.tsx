import { isValidElement, cloneElement, useEffect, useState } from 'react';
import type { ReactElement, ReactNode } from 'react';
import './primitives.css';

function TickingValue({ value }: { value: ReactNode }) {
  const metric = isValidElement(value) && typeof (value.props as { value?: unknown }).value === 'number';
  const raw = metric ? (value.props as { value: number }).value : typeof value === 'number' ? value : null;
  const [shown, setShown] = useState(raw ?? 0);
  useEffect(() => {
    if (raw == null) return;
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { setShown(raw); return; }
    const started = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const progress = Math.min(1, (now - started) / 620);
      setShown(raw * (1 - Math.pow(1 - progress, 3)));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [raw]);
  if (metric) return cloneElement(value as ReactElement<{ value: number }>, { value: shown });
  if (typeof value === 'number') return Number.isInteger(value) ? Math.round(shown) : shown.toFixed(2);
  return value;
}

export function StatTile({ label, value }: { label: string; value: ReactNode }) {
  return <div className="stat-tile"><span className="stat-label">{label}</span><strong className="stat-value"><TickingValue value={value} /></strong></div>;
}
