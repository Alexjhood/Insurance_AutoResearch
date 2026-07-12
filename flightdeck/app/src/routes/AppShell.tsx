import { useEffect, useState } from 'react';
import { Outlet, NavLink, useLocation, useNavigate, useParams } from 'react-router-dom';
import { Moon, RefreshCw, Sun } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { useIndex } from '../lib/data/queries';
import './AppShell.css';

type Theme = 'dark' | 'light';
export function AppShell() {
  const [theme, setTheme] = useState<Theme>(() => localStorage.getItem('flightdeck-theme') === 'light' ? 'light' : 'dark');
  const [rebuild, setRebuild] = useState<string | null>(null);
  const index = useIndex(); const queryClient = useQueryClient(); const navigate = useNavigate();
  const { orchId } = useParams(); const location = useLocation();
  const selected = orchId ?? location.pathname.match(/^\/o\/([^/]+)/)?.[1];
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('flightdeck-theme', theme); }, [theme]);
  async function runRebuild() {
    setRebuild('Starting rebuild…');
    try {
      const response = await fetch('/api/etl/rebuild', { method: 'POST' });
      if (!response.ok || !response.body) throw new Error('Rebuild request failed');
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending = '';
      while (true) {
        const { done, value } = await reader.read(); if (done) break; pending += decoder.decode(value, { stream: true });
        const events = pending.split('\n\n'); pending = events.pop() ?? '';
        for (const event of events) { const data = event.split('\n').find(line => line.startsWith('data: ')); if (data) { const parsed = JSON.parse(data.slice(6)) as {message?: string; refresh?: boolean}; setRebuild(parsed.message ?? 'Rebuilding…'); if (parsed.refresh) await queryClient.invalidateQueries({ queryKey: ['index'] }); } }
      }
    } catch (error) { setRebuild(error instanceof Error ? error.message : 'Rebuild failed'); return; }
    window.setTimeout(() => setRebuild(null), 2500);
  }
  return <div className="app-shell">
    <header className="topbar">
      <NavLink to="/" className="brand"><span className="brand-mark">FD</span><span>Flight Deck</span></NavLink>
      <label className="orch-switcher"><span className="sr-only">Orchestration</span><select value={selected ?? ''} onChange={e => navigate(e.target.value ? `/o/${e.target.value}` : '/')}><option value="">All orchestrations</option>{index.data?.orchestrations.map(o => <option key={o.orch_id} value={o.orch_id}>{o.alias || o.orch_id}</option>)}</select></label>
      <div className="top-actions">
        <button className="icon-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title={`Use ${theme === 'dark' ? 'light' : 'dark'} theme`} aria-label="Toggle theme">{theme === 'dark' ? <Sun /> : <Moon />}</button>
        <button className="command-button" onClick={runRebuild} disabled={Boolean(rebuild)}><RefreshCw className={rebuild ? 'spin' : ''} /> Rebuild</button>
      </div>
    </header>
    {rebuild && <div className="toast" role="status">{rebuild}</div>}
    {selected ? <div className="workspace"><aside className="side-nav"><NavLink end to={`/o/${selected}`}>Overview</NavLink><NavLink to={`/o/${selected}/journey`}>Journey</NavLink><NavLink to={`/o/${selected}/telemetry`}>Telemetry</NavLink></aside><div className="workspace-content"><Outlet /></div></div> : <Outlet />}
  </div>;
}
