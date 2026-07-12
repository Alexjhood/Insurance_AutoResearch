import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { DataProviderRoot, HttpProvider } from './lib/data/DataProvider';
import { AppShell } from './routes/AppShell';
import { Hangar } from './routes/Hangar/Hangar';
import { FileViewer } from './routes/FileViewer/FileViewer';
import { Placeholder } from './routes/Placeholder';
import './design/global.css';

const router = createBrowserRouter([{
  path: '/', element: <AppShell />, children: [
    { index: true, element: <Hangar /> },
    { path: 'compare', element: <Hangar leagueOnly /> },
    { path: 'o/:orchId', element: <Placeholder title="Campaign overview arrives in Phase 3" /> },
    { path: 'o/:orchId/journey', element: <Placeholder title="Journey arrives in Phase 3" /> },
    { path: 'o/:orchId/delegations/:dId', element: <Placeholder title="Delegation detail arrives in Phase 3" /> },
    { path: 'o/:orchId/telemetry', element: <Placeholder title="Resource analytics arrives in Phase 3" /> },
    { path: 'o/:orchId/files/*', element: <FileViewer /> },
  ],
}], { future: { v7_relativeSplatPath: true } });

const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 30_000, retry: 1 } } });
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><QueryClientProvider client={queryClient}><DataProviderRoot provider={new HttpProvider()}><RouterProvider router={router} future={{ v7_startTransition: true }} /></DataProviderRoot></QueryClientProvider></React.StrictMode>,
);
