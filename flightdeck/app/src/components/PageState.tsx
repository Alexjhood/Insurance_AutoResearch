export function PageState({ loading, error, children }: { loading: boolean; error: Error | null; children?: React.ReactNode }) {
  if (loading) return <main className="page page-state"><span className="state-pulse" /><p>Loading flight record...</p></main>;
  if (error) return <main className="page page-state error-state"><h1>Flight record unavailable</h1><p>{error.message}</p><button className="command-button" onClick={() => window.location.reload()}>Try again</button></main>;
  return <>{children}</>;
}
