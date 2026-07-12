import { createContext, useContext, type PropsWithChildren } from 'react';
import type { DelegationTelemetry, Snapshot, SnapshotIndex } from '../types';

export interface DataProvider {
  getIndex(): Promise<SnapshotIndex>;
  getSnapshot(id: string): Promise<Snapshot>;
  getTelemetry(id: string, delegationId: string): Promise<DelegationTelemetry>;
  getFileText(id: string, path: string): Promise<string>;
  getFileUrl(id: string, path: string): string;
}

async function checkedFetch(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status}: ${response.statusText}`);
  return response;
}

export class HttpProvider implements DataProvider {
  async getIndex() { return (await checkedFetch('/api/index')).json() as Promise<SnapshotIndex>; }
  async getSnapshot(id: string) { return (await checkedFetch(`/api/orchestrations/${encodeURIComponent(id)}`)).json() as Promise<Snapshot>; }
  async getTelemetry(id: string, delegationId: string) { return (await checkedFetch(`/api/orchestrations/${encodeURIComponent(id)}/telemetry/${encodeURIComponent(delegationId)}`)).json() as Promise<DelegationTelemetry>; }
  async getFileText(id: string, path: string) { return (await checkedFetch(this.getFileUrl(id, path))).text(); }
  getFileUrl(id: string, path: string) { return `/api/orchestrations/${encodeURIComponent(id)}/files/${path.split('/').map(encodeURIComponent).join('/')}`; }
}

export class EmbeddedProvider implements DataProvider {
  private unavailable(): never { throw new Error('Embedded snapshots are added in Phase 5'); }
  async getIndex(): Promise<SnapshotIndex> { return this.unavailable(); }
  async getSnapshot(_id: string): Promise<Snapshot> { return this.unavailable(); }
  async getTelemetry(_id: string, _delegationId: string): Promise<DelegationTelemetry> { return this.unavailable(); }
  async getFileText(_id: string, _path: string): Promise<string> { return this.unavailable(); }
  getFileUrl(_id: string, _path: string): string { return this.unavailable(); }
}

const Context = createContext<DataProvider | null>(null);
export function DataProviderRoot({ provider, children }: PropsWithChildren<{provider: DataProvider}>) { return <Context.Provider value={provider}>{children}</Context.Provider>; }
export function useDataProvider(): DataProvider { const provider = useContext(Context); if (!provider) throw new Error('DataProvider missing'); return provider; }
