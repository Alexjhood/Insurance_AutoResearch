import './primitives.css';
export function StatTile({ label, value }: {label: string; value: React.ReactNode}) { return <div className="stat-tile"><span className="stat-label">{label}</span><strong className="stat-value">{value}</strong></div>; }
