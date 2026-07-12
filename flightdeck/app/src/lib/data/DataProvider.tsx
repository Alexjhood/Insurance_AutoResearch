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
  private blobs = new Map<string, string>();

  private read<T>(id: string): T {
    const element = document.getElementById(id);
    if (!element?.textContent) throw new Error(`Static export is missing ${id}`);
    return JSON.parse(element.textContent) as T;
  }

  async getIndex(): Promise<SnapshotIndex> { return this.read('fd-embedded-index'); }
  async getSnapshot(id: string): Promise<Snapshot> { return this.read(`fd-embedded-snapshot-${id}`); }
  async getTelemetry(id: string, delegationId: string): Promise<DelegationTelemetry> { return this.read(`fd-embedded-telemetry-${id}-${delegationId}`); }
  async getFileText(id: string, path: string): Promise<string> {
    const files = this.read<Record<string, string>>(`fd-embedded-files-${id}`);
    if (!(path in files)) throw new Error('This file is larger than 512 KB and was not included in the static export.');
    return files[path];
  }
  getFileUrl(id: string, path: string): string {
    const key = `${id}/${path}`;
    const existing = this.blobs.get(key);
    if (existing) return existing;
    const files = this.read<Record<string, string>>(`fd-embedded-files-${id}`);
    if (!(path in files)) throw new Error('This file was not included in the static export.');
    const url = URL.createObjectURL(new Blob([files[path]], { type: 'text/plain;charset=utf-8' }));
    this.blobs.set(key, url);
    return url;
  }
}

export function isEmbeddedExport(): boolean { return Boolean(document.getElementById('fd-embedded-meta')); }

const Context = createContext<DataProvider | null>(null);
export function DataProviderRoot({ provider, children }: PropsWithChildren<{provider: DataProvider}>) { return <Context.Provider value={provider}>{children}</Context.Provider>; }
export function useDataProvider(): DataProvider { const provider = useContext(Context); if (!provider) throw new Error('DataProvider missing'); return provider; }
