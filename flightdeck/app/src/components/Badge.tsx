import './primitives.css';
const glyphs = { promote: '▲', local_promote: '◭', reject: '○', distress: '✕', takeover: '⚑', reflection: '◆', orchestrator: '▣', seed: '⇢', baseline: '┄' } as const;
export type BadgeKind = keyof typeof glyphs;
export function Badge({ kind, children }: {kind: BadgeKind; children: React.ReactNode}) { return <span className={`badge badge-${kind}`}>{glyphs[kind]} {children}</span>; }
