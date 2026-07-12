import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import type { Delegation, Experiment } from '../lib/types';
import { useExperimentHover } from '../lib/experimentHover';
import {
  ExhibitShell, LegendItem, useMeasure, useReducedMotion,
  outcomeOf, OUTCOME_GLYPH, OUTCOME_LABEL, OUTCOME_VAR, delegationVar,
} from './shared';
import './exhibits.css';

export interface ResearchTreeProps { orchId: string; experiments: Experiment[]; delegations: Delegation[] }

interface TreeNode { e: Experiment; children: TreeNode[]; depth: number; x: number }

function journeyHash(e: Experiment): string {
  if (e.delegation_id == null) return '#finale';
  if (e.is_seed || e.is_baseline) return `#${e.delegation_id}`;
  return `#${e.delegation_id}-x${e.cycle ?? e.seq}`;
}

function nodeLabel(e: Experiment): string {
  if (e.is_seed) return `seed ${e.delegation_id ?? 'playoff'}`;
  const name = e.name.replace(/^orchestration_/, '');
  return name.length > 22 ? `${name.slice(0, 21)}…` : name;
}

/**
 * The shape of the search: experiments as nodes, scientific parentage as solid
 * edges, orchestrator seed transfers as dashed edges bridging delegations.
 * Tidy layered layout — leaves get sequential x slots, parents centre on their
 * children, depth is distance from the campaign root.
 */
export function ResearchTree({ orchId, experiments, delegations }: ResearchTreeProps) {
  const [wrapRef, width] = useMeasure<HTMLDivElement>();
  const { showExperiment, moveExperiment, clearExperiment, classNameFor } = useExperimentHover();
  const navigate = useNavigate();
  const reduced = useReducedMotion();
  const w = Math.max(width, 640);

  const model = useMemo(() => {
    const byId = new Map(experiments.map(e => [e.experiment_id, e]));
    // seed-transfer edges: delegation seed experiment ← brief.seed_champion source
    const seedSource = new Map<string, string>();
    for (const d of delegations) {
      const src = d.brief.seed_champion?.experiment_id;
      if (!src || !byId.has(src)) continue;
      const seed = experiments.find(e => e.delegation_id === d.delegation_id && e.is_seed);
      if (seed) seedSource.set(seed.experiment_id, src);
    }
    const parentOf = (e: Experiment): string | null => {
      if (seedSource.has(e.experiment_id)) return seedSource.get(e.experiment_id)!;
      if (e.parent_experiment_id && byId.has(e.parent_experiment_id)) return e.parent_experiment_id;
      return null;
    };
    const nodes = new Map<string, TreeNode>(experiments.map(e => [e.experiment_id, { e, children: [], depth: 0, x: 0 }]));
    const roots: TreeNode[] = [];
    for (const n of nodes.values()) {
      const p = parentOf(n.e);
      if (p) nodes.get(p)!.children.push(n); else roots.push(n);
    }
    const bySeq = (a: TreeNode, b: TreeNode) => a.e.seq - b.e.seq;
    roots.sort(bySeq);
    let slot = 0, maxDepth = 0;
    const place = (n: TreeNode, depth: number) => {
      n.depth = depth; maxDepth = Math.max(maxDepth, depth);
      n.children.sort(bySeq);
      if (!n.children.length) { n.x = slot++; return; }
      n.children.forEach(c => place(c, depth + 1));
      n.x = (n.children[0].x + n.children[n.children.length - 1].x) / 2;
    };
    roots.forEach(r => place(r, 0));
    const edges = [...nodes.values()].flatMap(n => n.children.map(c => ({
      from: n, to: c, isSeed: seedSource.has(c.e.experiment_id),
    })));
    return { nodes: [...nodes.values()], edges, slots: Math.max(slot, 1), maxDepth };
  }, [experiments, delegations]);

  const M = { l: 28, r: 28, t: 30, b: 26 };
  const H = M.t + M.b + (model.maxDepth + 1) * 62;
  const px = (x: number) => M.l + (model.slots === 1 ? 0.5 : x / (model.slots - 1)) * (w - M.l - M.r);
  const py = (depth: number) => M.t + depth * 62;

  const table = (
    <>
      <thead><tr><th>Experiment</th><th>Delegation</th><th>Research line</th><th>Parent</th><th>Decision</th></tr></thead>
      <tbody>
        {experiments.map(e => {
          const o = outcomeOf(e);
          return (
            <tr key={e.experiment_id}>
              <td className="mono">{e.name}</td>
              <td><span style={{ color: `var(${delegationVar(e.delegation_id)})` }}>{e.delegation_id ?? 'playoff'}</span></td>
              <td className="mono">{e.research_line_id ?? '—'}</td>
              <td className="mono">{e.parent_experiment_id ? e.parent_experiment_id.replace(/^\d+T\d+Z_/, '') : '—'}</td>
              <td><span style={{ color: `var(${OUTCOME_VAR[o]})` }}>{OUTCOME_GLYPH[o]} {OUTCOME_LABEL[o]}</span></td>
            </tr>
          );
        })}
      </tbody>
    </>
  );

  return (
    <ExhibitShell
      eyebrow="Research tree" title="The shape of the search"
      note={`${model.edges.filter(e => e.isSeed).length} seed transfers bridge the delegations`}
      legend={<>
        <LegendItem colorVar="--c-promote" glyph="▲">promote</LegendItem>
        <LegendItem colorVar="--c-localpromote" glyph="◭">local promote</LegendItem>
        <LegendItem colorVar="--c-reject" glyph="○">reject</LegendItem>
        <LegendItem colorVar="--c-distress" glyph="✕">forfeited</LegendItem>
        <LegendItem colorVar="--c-baseline" glyph="⇢">seed transfer (dashed)</LegendItem>
        <LegendItem colorVar="--c-orchestrator" glyph="◈">playoff replay</LegendItem>
      </>}
      table={table} tableCaption="Experiment parentage and research lines"
    >
      <div className="chart-well" ref={wrapRef}>
        <svg className={`research-tree ${reduced ? '' : 'animate-in'}`} width="100%" height={H} viewBox={`0 0 ${w} ${H}`} role="img"
          aria-label="Research tree: experiment parentage across the campaign, with seed transfers bridging delegations">
          {model.edges.map(({ from, to, isSeed }) => {
            const x0 = px(from.x), y0 = py(from.depth) + 10, x1 = px(to.x), y1 = py(to.depth) - 14;
            return (
              <path key={to.e.experiment_id} className={isSeed ? 'tree-edge seed' : 'tree-edge'}
                style={isSeed ? { stroke: `var(${delegationVar(to.e.delegation_id)})` } : undefined}
                d={`M ${x0} ${y0} C ${x0} ${(y0 + y1) / 2}, ${x1} ${(y0 + y1) / 2}, ${x1} ${y1}`} />
            );
          })}
          {model.nodes.map(n => {
            const o = outcomeOf(n.e);
            return (
              <g key={n.e.experiment_id} className={`tree-node ${classNameFor(n.e.experiment_id)}`} tabIndex={0} role="link"
                aria-label={`${n.e.name}, ${OUTCOME_LABEL[o]} — open journey card`}
                onMouseEnter={ev => showExperiment(ev, n.e)} onMouseMove={moveExperiment} onMouseLeave={clearExperiment}
                onFocus={() => showExperiment({ clientX: 80, clientY: 120 }, n.e)} onBlur={clearExperiment}
                onClick={() => { clearExperiment(); navigate(`/o/${orchId}/journey${journeyHash(n.e)}`); }}
                onKeyDown={ev => { if (ev.key === 'Enter') { clearExperiment(); navigate(`/o/${orchId}/journey${journeyHash(n.e)}`); } }}>
                <circle cx={px(n.x)} cy={py(n.depth)} r={13} className="tree-halo"
                  style={{ fill: `var(${delegationVar(n.e.delegation_id)})` }} />
                <text x={px(n.x)} y={py(n.depth) + 5} textAnchor="middle" className="tree-glyph"
                  style={{ fill: `var(${OUTCOME_VAR[o]})` }}>{OUTCOME_GLYPH[o]}</text>
                <text x={px(n.x)} y={py(n.depth) + 26 + (Math.round(n.x) % 2) * 12} textAnchor="middle" className="tree-label">
                  {nodeLabel(n.e)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </ExhibitShell>
  );
}
