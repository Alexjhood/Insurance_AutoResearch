export function PageState({ loading, error, children }: { loading: boolean; error: Error | null; children?: React.ReactNode }) {
  if (loading) return <main className="page"><p>Loading flight record…</p></main>;
  if (error) return <main className="page error-state">Unable to load flight record: {error.message}</main>;
  return <>{children}</>;
}
