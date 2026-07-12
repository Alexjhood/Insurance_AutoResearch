import { useEffect, useMemo, useRef, useState } from 'react';
import { FileText, FlaskConical, GitBranch, RefreshCw, Search, Trophy, X } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { Snapshot, SnapshotIndex } from '../lib/types';

interface CommandItem { id: string; label: string; detail: string; group: string; icon: LucideIcon; run(): void }
interface CommandPaletteProps {
  open: boolean;
  onClose(): void;
  index: SnapshotIndex | undefined;
  snapshot: Snapshot | undefined;
  embedded: boolean;
  navigate(path: string): void;
  rebuild(): void;
}

function fuzzyScore(query: string, text: string) {
  const needle = query.toLowerCase().trim();
  if (!needle) return 0;
  const haystack = text.toLowerCase();
  const direct = haystack.indexOf(needle);
  if (direct >= 0) return 1000 - direct;
  let cursor = 0, score = 0, streak = 0;
  for (const char of needle) {
    const found = haystack.indexOf(char, cursor);
    if (found < 0) return -1;
    streak = found === cursor ? streak + 1 : 0;
    score += 10 + streak * 4 - Math.min(found - cursor, 8);
    cursor = found + 1;
  }
  return score;
}

export function CommandPalette({ open, onClose, index, snapshot, embedded, navigate, rebuild }: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const orchId = snapshot?.campaign.orch_id;
  const items = useMemo<CommandItem[]>(() => [
    ...(index?.orchestrations.map(campaign => ({ id: `campaign-${campaign.orch_id}`, label: campaign.alias || campaign.orch_id, detail: `${campaign.dataset} · ${campaign.target_mode}`, group: 'Campaigns', icon: Trophy, run: () => navigate(`/o/${campaign.orch_id}`) })) ?? []),
    ...(snapshot?.delegations.map(delegation => ({ id: `delegation-${delegation.delegation_id}`, label: `${delegation.delegation_id} · ${delegation.brief.name ?? delegation.backend}`, detail: delegation.brief.direction, group: 'Delegations', icon: GitBranch, run: () => navigate(`/o/${orchId}/delegations/${delegation.delegation_id}`) })) ?? []),
    ...(snapshot?.experiments.map(experiment => ({ id: `experiment-${experiment.experiment_id}`, label: experiment.name, detail: `${experiment.delegation_id ?? 'playoff'} · cycle ${experiment.cycle ?? 'seed'}`, group: 'Experiments', icon: FlaskConical, run: () => navigate(`/o/${orchId}/journey#${experiment.delegation_id == null ? 'finale' : experiment.is_seed || experiment.is_baseline ? experiment.delegation_id : `${experiment.delegation_id}-x${experiment.cycle ?? experiment.seq}`}`) })) ?? []),
    ...(snapshot?.files.map(file => ({ id: `file-${file.path}`, label: file.path.split('/').at(-1) ?? file.path, detail: file.path, group: 'Files', icon: FileText, run: () => navigate(`/o/${orchId}/files/${file.path}`) })) ?? []),
    ...(!embedded ? [{ id: 'action-rebuild', label: 'Rebuild snapshots', detail: 'Run incremental Flight Deck ETL', group: 'Actions', icon: RefreshCw, run: rebuild }] : []),
    ...(snapshot?.playoff?.report_md ? [{ id: 'action-playoff', label: 'Open playoff evidence', detail: snapshot.playoff.report_md, group: 'Actions', icon: Trophy, run: () => navigate(`/o/${orchId}/files/${snapshot.playoff!.report_md}`) }] : []),
  ], [index, snapshot, orchId, embedded, navigate, rebuild]);
  const results = useMemo(() => items.map(item => ({ item, score: fuzzyScore(query, `${item.label} ${item.detail} ${item.group}`) })).filter(x => x.score >= 0).sort((a, b) => b.score - a.score).slice(0, 12).map(x => x.item), [items, query]);

  useEffect(() => { if (open) { setQuery(''); setActive(0); requestAnimationFrame(() => inputRef.current?.focus()); } }, [open]);
  useEffect(() => setActive(0), [query]);
  if (!open) return null;
  const choose = (item: CommandItem) => { item.run(); onClose(); };
  return <div className="command-scrim" role="presentation" onMouseDown={event => event.target === event.currentTarget && onClose()}>
    <section className="command-palette" role="dialog" aria-modal="true" aria-label="Flight Deck command palette" onKeyDown={event => {
      if (event.key === 'Escape') onClose();
      if (event.key === 'ArrowDown') { event.preventDefault(); setActive(value => Math.min(value + 1, results.length - 1)); }
      if (event.key === 'ArrowUp') { event.preventDefault(); setActive(value => Math.max(value - 1, 0)); }
      if (event.key === 'Enter' && results[active]) { event.preventDefault(); choose(results[active]); }
    }}>
      <header><Search aria-hidden="true" /><input ref={inputRef} value={query} onChange={event => setQuery(event.target.value)} placeholder="Jump to campaign, delegation, experiment, file…" aria-label="Search commands" /><button type="button" onClick={onClose} aria-label="Close command palette"><X /></button></header>
      <div className="command-results" role="listbox">{results.length ? results.map((item, indexValue) => { const Icon = item.icon; return <button type="button" role="option" aria-selected={indexValue === active} className={indexValue === active ? 'active' : ''} key={item.id} onMouseEnter={() => setActive(indexValue)} onClick={() => choose(item)}><Icon /><span><b>{item.label}</b><small>{item.detail}</small></span><em>{item.group}</em></button>; }) : <p>No matching flight record.</p>}</div>
      <footer><span>↑↓ navigate</span><span>↵ open</span><span>esc close</span></footer>
    </section>
  </div>;
}
