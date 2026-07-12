export function compact(value: number) { return Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(value); }
export function duration(minutes: number | null) { if (minutes == null) return '—'; const h = Math.floor(minutes / 60); const m = Math.round(minutes % 60); return h ? `${h}h ${m}m` : `${m}m`; }
export function dateTime(value: string | null) { return value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—'; }
export function totalTokens(tokens: { input: number; output: number; reasoning: number }) { return tokens.input + tokens.output + tokens.reasoning; }
export function decisionKind(decision: string | null | undefined, status?: string) { if (decision === 'promote' || decision === 'local_promote' || decision === 'reject') return decision; return status === 'completed' ? 'distress' : 'takeover'; }
