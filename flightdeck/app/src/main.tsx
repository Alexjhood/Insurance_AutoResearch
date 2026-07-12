import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { DataProviderRoot, HttpProvider } from './lib/data/DataProvider';
import { AppShell } from './routes/AppShell';
import { Hangar } from './routes/Hangar/Hangar';
import { FileViewer } from './routes/FileViewer/FileViewer';
import { Overview } from './routes/Overview/Overview';
import { Journey } from './routes/Journey/Journey';
import { DelegationDetail } from './routes/Delegation/Delegation';
import { Telemetry } from './routes/Telemetry/Telemetry';
import './design/global.css';

const router = createBrowserRouter([{
  path: '/', element: <AppShell />, children: [
    { index: true, element: <Hangar /> },
    { path: 'compare', element: <Hangar leagueOnly /> },
    { path: 'o/:orchId', element: <Overview /> },
    { path: 'o/:orchId/journey', element: <Journey /> },
    { path: 'o/:orchId/delegations/:dId', element: <DelegationDetail /> },
    { path: 'o/:orchId/telemetry', element: <Telemetry /> },
    { path: 'o/:orchId/files/*', element: <FileViewer /> },
  ],
}], { future: { v7_relativeSplatPath: true } });

const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 30_000, retry: 1 } } });
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><QueryClientProvider client={queryClient}><DataProviderRoot provider={new HttpProvider()}><RouterProvider router={router} future={{ v7_startTransition: true }} /></DataProviderRoot></QueryClientProvider></React.StrictMode>,
);
