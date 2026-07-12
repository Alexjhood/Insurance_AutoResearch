import { useEffect, useState } from 'react';
import { Outlet, NavLink, useLocation, useNavigate, useParams } from 'react-router-dom';
import { Command, Moon, RefreshCw, Sun } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { useIndex, useSnapshot } from '../lib/data/queries';
import { isEmbeddedExport } from '../lib/data/DataProvider';
import { CommandPalette } from '../components/CommandPalette';
import './AppShell.css';

type Theme = 'dark' | 'light';
export function AppShell() {
  const embedded = isEmbeddedExport();
  const [theme, setTheme] = useState<Theme>(() => localStorage.getItem('flightdeck-theme') === 'light' ? 'light' : 'dark');
  const [rebuild, setRebuild] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const index = useIndex(); const queryClient = useQueryClient(); const navigate = useNavigate();
  const { orchId } = useParams(); const location = useLocation();
  const selected = orchId ?? location.pathname.match(/^\/o\/([^/]+)/)?.[1];
  const snapshot = useSnapshot(selected);
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('flightdeck-theme', theme); }, [theme]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); setPaletteOpen(value => !value); } };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
  async function runRebuild() {
    setRebuild('Starting rebuild…');
    try {
      const response = await fetch('/api/etl/rebuild', { method: 'POST' });
      if (!response.ok || !response.body) throw new Error('Rebuild request failed');
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending = '';
      while (true) {
        const { done, value } = await reader.read(); if (done) break; pending += decoder.decode(value, { stream: true });
        const events = pending.split('\n\n'); pending = events.pop() ?? '';
        for (const event of events) { const data = event.split('\n').find(line => line.startsWith('data: ')); if (data) { const parsed = JSON.parse(data.slice(6)) as {message?: string; refresh?: boolean}; setRebuild(parsed.message ?? 'Rebuilding…'); if (parsed.refresh) await queryClient.invalidateQueries(); } }
      }
    } catch (error) { setRebuild(error instanceof Error ? error.message : 'Rebuild failed'); return; }
    window.setTimeout(() => setRebuild(null), 2500);
  }
  return <div className="app-shell">
    {embedded && <div className="export-ribbon" role="status">Static export <span>Built {document.getElementById('fd-embedded-meta')?.getAttribute('data-built-at')}</span></div>}
    <header className="topbar">
      <NavLink to="/" className="brand"><span className="brand-mark">FD</span><span>Flight Deck</span></NavLink>
      <label className="orch-switcher"><span className="sr-only">Orchestration</span><select value={selected ?? ''} onChange={e => navigate(e.target.value ? `/o/${e.target.value}` : '/')}><option value="">All orchestrations</option>{index.data?.orchestrations.map(o => <option key={o.orch_id} value={o.orch_id}>{o.alias || o.orch_id}</option>)}</select></label>
      <div className="top-actions">
        <button className="command-button palette-trigger" onClick={() => setPaletteOpen(true)}><Command /> Search <kbd>⌘K</kbd></button>
        <button className="icon-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title={`Use ${theme === 'dark' ? 'light' : 'dark'} theme`} aria-label="Toggle theme">{theme === 'dark' ? <Sun /> : <Moon />}</button>
        {!embedded && <button className="command-button" onClick={runRebuild} disabled={Boolean(rebuild)}><RefreshCw className={rebuild ? 'spin' : ''} /> Rebuild</button>}
      </div>
    </header>
    {rebuild && <div className="toast" role="status">{rebuild}</div>}
    <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} index={index.data} snapshot={snapshot.data} embedded={embedded} navigate={navigate} rebuild={() => { void runRebuild(); }} />
    {selected ? <div className="workspace"><aside className="side-nav"><NavLink end to={`/o/${selected}`}>Overview</NavLink><NavLink to={`/o/${selected}/journey`}>Journey</NavLink><NavLink to={`/o/${selected}/telemetry`}>Telemetry</NavLink></aside><div className="workspace-content"><Outlet /></div></div> : <Outlet />}
  </div>;
}
