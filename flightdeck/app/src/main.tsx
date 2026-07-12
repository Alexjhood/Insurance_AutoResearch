import React, { Suspense, lazy } from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createBrowserRouter, createHashRouter, RouterProvider } from 'react-router-dom';
import { DataProviderRoot, EmbeddedProvider, HttpProvider, isEmbeddedExport } from './lib/data/DataProvider';
import { AppShell } from './routes/AppShell';
import './design/global.css';

const Hangar = lazy(() => import('./routes/Hangar/Hangar').then(m => ({ default: m.Hangar })));
const Overview = lazy(() => import('./routes/Overview/Overview').then(m => ({ default: m.Overview })));
const Journey = lazy(() => import('./routes/Journey/Journey').then(m => ({ default: m.Journey })));
const DelegationDetail = lazy(() => import('./routes/Delegation/Delegation').then(m => ({ default: m.DelegationDetail })));
const Telemetry = lazy(() => import('./routes/Telemetry/Telemetry').then(m => ({ default: m.Telemetry })));
const FileViewer = lazy(() => import('./routes/FileViewer/FileViewer').then(m => ({ default: m.FileViewer })));
const loading = <main className="page page-state"><span className="state-pulse" />Loading flight record...</main>;
const embedded = isEmbeddedExport();
const routes = [{
  path: '/', element: <AppShell />, children: [
    { index: true, element: <Suspense fallback={loading}><Hangar /></Suspense> },
    { path: 'compare', element: <Suspense fallback={loading}><Hangar leagueOnly /></Suspense> },
    { path: 'o/:orchId', element: <Suspense fallback={loading}><Overview /></Suspense> },
    { path: 'o/:orchId/journey', element: <Suspense fallback={loading}><Journey /></Suspense> },
    { path: 'o/:orchId/delegations/:dId', element: <Suspense fallback={loading}><DelegationDetail /></Suspense> },
    { path: 'o/:orchId/telemetry', element: <Suspense fallback={loading}><Telemetry /></Suspense> },
    { path: 'o/:orchId/files/*', element: <Suspense fallback={loading}><FileViewer /></Suspense> },
    { path: '*', element: <main className="page page-state"><h1>Flight record not found</h1><a href={embedded ? '#/' : '/'}>Return to Hangar</a></main> },
  ],
}];
const router = (embedded ? createHashRouter : createBrowserRouter)(routes, { future: { v7_relativeSplatPath: true } });

const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 30_000, retry: 1 } } });
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><QueryClientProvider client={queryClient}><DataProviderRoot provider={embedded ? new EmbeddedProvider() : new HttpProvider()}><RouterProvider router={router} future={{ v7_startTransition: true }} /></DataProviderRoot></QueryClientProvider></React.StrictMode>,
);
