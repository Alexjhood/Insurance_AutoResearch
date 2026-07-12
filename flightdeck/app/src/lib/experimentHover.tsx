import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type { PropsWithChildren } from 'react';
import type { Experiment } from './types';

interface HoverPoint { clientX: number; clientY: number }
interface ExperimentHoverValue {
  hoveredExperimentId: string | null;
  showExperiment(event: HoverPoint, experiment: Experiment): void;
  moveExperiment(event: HoverPoint): void;
  clearExperiment(): void;
  classNameFor(experimentId: string): string;
}

interface HoverState { x: number; y: number; experiment: Experiment }
const ExperimentHoverContext = createContext<ExperimentHoverValue | null>(null);

function outcome(experiment: Experiment) {
  if (experiment.is_baseline) return '◇ baseline';
  if (experiment.is_seed) return '⇢ seed transfer';
  if (experiment.delegation_id == null) return '◈ playoff replay';
  const decision = experiment.comparison?.decision;
  return decision === 'promote' ? '▲ promote' : decision === 'local_promote' ? '◭ local promote' : decision === 'reject' ? '○ reject' : '✕ forfeited / undecided';
}

export function ExperimentHoverProvider({ children }: PropsWithChildren) {
  const [hover, setHover] = useState<HoverState | null>(null);
  const position = useCallback((event: HoverPoint) => ({
    x: Math.min(event.clientX + 14, window.innerWidth - 316),
    y: Math.min(event.clientY + 14, window.innerHeight - 150),
  }), []);
  const showExperiment = useCallback((event: HoverPoint, experiment: Experiment) => setHover({ ...position(event), experiment }), [position]);
  const moveExperiment = useCallback((event: HoverPoint) => setHover(current => current ? { ...current, ...position(event) } : null), [position]);
  const clearExperiment = useCallback(() => setHover(null), []);
  const value = useMemo<ExperimentHoverValue>(() => ({
    hoveredExperimentId: hover?.experiment.experiment_id ?? null,
    showExperiment, moveExperiment, clearExperiment,
    classNameFor: experimentId => hover == null ? '' : hover.experiment.experiment_id === experimentId ? 'experiment-linked-active' : 'experiment-linked-dimmed',
  }), [hover, showExperiment, moveExperiment, clearExperiment]);
  const e = hover?.experiment;
  return <ExperimentHoverContext.Provider value={value}>
    {children}
    {e && <div className="exhibit-tip shared-experiment-tip" role="tooltip" style={{ left: hover.x, top: hover.y }}>
      <div className="tip-title">{e.name}</div>
      <div className="tip-meta mono">{e.delegation_id ?? 'playoff'}{e.cycle != null ? ` · cycle ${e.cycle}` : ''} · {e.model_family}</div>
      <div className="tip-meta mono">{outcome(e)}{e.metrics.gini_weighted != null ? ` · gini ${e.metrics.gini_weighted.toFixed(4)}` : ''}{e.lift.vs_then_champion != null ? ` · lift ${e.lift.vs_then_champion >= 0 ? '+' : '−'}${Math.abs(e.lift.vs_then_champion).toFixed(4)}` : ''}</div>
      {e.proposal.change_summary && <p className="tip-body">{e.proposal.change_summary}</p>}
      <div className="tip-hint">linked across campaign exhibits</div>
    </div>}
  </ExperimentHoverContext.Provider>;
}

export function useExperimentHover() {
  const value = useContext(ExperimentHoverContext);
  if (!value) throw new Error('useExperimentHover must be used within ExperimentHoverProvider');
  return value;
}
