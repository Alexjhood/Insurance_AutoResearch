import { useQuery } from '@tanstack/react-query';
import { useDataProvider } from './DataProvider';

export function useIndex() { const data = useDataProvider(); return useQuery({ queryKey: ['index'], queryFn: () => data.getIndex() }); }
export function useSnapshot(id: string | undefined) { const data = useDataProvider(); return useQuery({ queryKey: ['snapshot', id], queryFn: () => data.getSnapshot(id!), enabled: Boolean(id) }); }
export function useTelemetry(id: string | undefined, delegationId: string | undefined) { const data = useDataProvider(); return useQuery({ queryKey: ['telemetry', id, delegationId], queryFn: () => data.getTelemetry(id!, delegationId!), enabled: Boolean(id && delegationId) }); }
export function useFile(id: string | undefined, path: string | undefined) { const data = useDataProvider(); return useQuery({ queryKey: ['file', id, path], queryFn: () => data.getFileText(id!, path!), enabled: Boolean(id && path) }); }
