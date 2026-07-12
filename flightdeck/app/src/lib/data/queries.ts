import { useQuery } from '@tanstack/react-query';
import { useDataProvider } from './DataProvider';

export function useIndex() { const data = useDataProvider(); return useQuery({ queryKey: ['index'], queryFn: () => data.getIndex() }); }
export function useSnapshot(id: string | undefined) { const data = useDataProvider(); return useQuery({ queryKey: ['snapshot', id], queryFn: () => data.getSnapshot(id!), enabled: Boolean(id) }); }
export function useFile(id: string | undefined, path: string | undefined) { const data = useDataProvider(); return useQuery({ queryKey: ['file', id, path], queryFn: () => data.getFileText(id!, path!), enabled: Boolean(id && path) }); }
