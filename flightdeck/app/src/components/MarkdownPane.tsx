import DOMPurify from 'dompurify'; import { marked } from 'marked'; import './primitives.css';
export function MarkdownPane({ source }: {source: string}) { const html = DOMPurify.sanitize(marked.parse(source, { async: false }) as string); return <article className="markdown-pane" dangerouslySetInnerHTML={{ __html: html }} />; }
