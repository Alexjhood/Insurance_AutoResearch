import { useParams } from 'react-router-dom';
export function Placeholder({ title }: { title: string }) { const { orchId } = useParams(); return <main className="page"><span className="eyebrow mono">{orchId}</span><h1>{title}</h1></main>; }
