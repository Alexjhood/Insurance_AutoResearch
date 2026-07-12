export function compact(value: number) { return Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(value); }
export function duration(minutes: number | null) { if (minutes == null) return '—'; const h = Math.floor(minutes / 60); const m = Math.round(minutes % 60); return h ? `${h}h ${m}m` : `${m}m`; }
export function dateTime(value: string | null) { return value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—'; }
export function totalTokens(tokens: { input: number; output: number; reasoning: number }) { return tokens.input + tokens.output + tokens.reasoning; }
export function decisionKind(decision: string | null | undefined, status?: string) { if (decision === 'promote' || decision === 'local_promote' || decision === 'reject') return decision; return status === 'completed' ? 'distress' : 'takeover'; }

interface RecipeShape { estimator?: string; objective?: string; encoding?: string }
/** One-line human descriptor for a recipe, e.g. "hist_gbm · poisson · ordinal". */
export function recipeSummary(recipe: unknown): string | null {
  if (!recipe || typeof recipe !== 'object') return null;
  const r = recipe as RecipeShape;
  const parts = [r.estimator, r.objective, r.encoding === 'native_categorical' ? 'native cat' : r.encoding?.replace('_', ' ')].filter(Boolean);
  return parts.length ? parts.join(' · ') : null;
}
/** Seed/consolidation experiments carry no recipe; their name embeds the source experiment's
 *  name as a suffix (e.g. "…_seed_d05_hist_gbm_poisson_ordinal"). Resolve to the longest
 *  recipe-bearing experiment whose name the target's name ends with. */
export function resolveRecipe(experiments: { name: string; recipe: unknown | null }[], exp: { name: string; recipe: unknown | null } | undefined): unknown | null {
  if (!exp) return null;
  if (exp.recipe) return exp.recipe;
  let best: { name: string; recipe: unknown | null } | null = null;
  for (const source of experiments) {
    if (!source.recipe || !exp.name.endsWith(source.name)) continue;
    if (!best || source.name.length > best.name.length) best = source;
  }
  return best?.recipe ?? null;
}

export function orchestratorLabel(value: { provider: string; model: string; effort: string | null } | undefined, legacy = ''): string {
  if (!value?.model) return legacy;
  return `${value.model}${value.effort ? ` · ${value.effort}` : ''}`;
}

export function championDescriptor(experiments: { name: string; model_family: string; recipe: unknown | null }[], exp: { name: string; model_family: string; recipe: unknown | null } | undefined, fallback = 'Campaign champion'): string {
  const recipe = recipeSummary(resolveRecipe(experiments, exp));
  if (recipe) return recipe;
  if (!exp) return fallback;
  const raw = exp.name
    .replace(/^\d{8}T\d{6}Z_/, '')
    .replace(/^orchestration_(?:challenger_\d+_[^_]+_|seed_[^_]+_)/, '')
    .replace(/^orchestration_delegation_seed_[^_]+_/, '');
  const blend = raw.match(/(.+)_blend_(\d+)_(\d+)$/);
  if (blend) return `${blend[1].replace(/_/g, ' + ')} · ${blend[2]}/${blend[3]} blend`;
  return raw.replace(/_/g, ' ');
}

function flattenRecipe(value: unknown, prefix = '', output: Record<string, string> = {}): Record<string, string> {
  if (value == null) return output;
  if (Array.isArray(value)) { output[prefix] = value.join(', '); return output; }
  if (typeof value !== 'object') { output[prefix] = String(value); return output; }
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    flattenRecipe(child, prefix ? `${prefix}.${key}` : key, output);
  }
  return output;
}

/** Scalar recipe changes computed from the normalized config snapshots carried by experiments. */
export function recipeDiff(experiments: { experiment_id: string; name: string; recipe: unknown | null }[], experiment: { parent_experiment_id: string | null; name: string; recipe: unknown | null }): string[] {
  const current = flattenRecipe(resolveRecipe(experiments, experiment));
  if (!Object.keys(current).length) return [];
  const parent = experiments.find(item => item.experiment_id === experiment.parent_experiment_id);
  const previous = flattenRecipe(resolveRecipe(experiments, parent));
  if (!Object.keys(previous).length) return [];
  const keys = [...new Set([...Object.keys(previous), ...Object.keys(current)])].sort();
  return keys.filter(key => previous[key] !== current[key]).map(key => {
    const label = key.replace(/^params\./, '').replace(/^stages\./, '').replaceAll('.', ' · ').replaceAll('_', ' ');
    return `${label} ${previous[key] ?? '∅'}→${current[key] ?? '∅'}`;
  });
}
