import type { ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type MarkdownDocumentProps = {
  markdown: string;
  sourcePath: string;
};

export type MarkdownSection = {
  id: string;
  level: 2 | 3;
  title: string;
};

export const DOC_ROUTES: Record<string, string> = {
  'README.md': '#/readme',
  'AGENTS.md': '#/maintainer',
  'docs/maintainer-guide.md': '#/maintainer',
  'docs/setup-guide.md': '#/setup',
  'docs/user-guide.md': '#/guide'
};

const GITHUB_BLOB_ROOT = 'https://github.com/jasencarroll/vibe-rag/blob/main/';

function normalizeRepoPath(path: string): string {
  const parts = path.split('/');
  const output: string[] = [];
  for (const part of parts) {
    if (!part || part === '.') continue;
    if (part === '..') {
      output.pop();
      continue;
    }
    output.push(part);
  }
  return output.join('/');
}

function flattenText(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(flattenText).join('');
  }
  if (node && typeof node === 'object' && 'props' in node) {
    return flattenText((node as { props?: { children?: ReactNode } }).props?.children ?? '');
  }
  return '';
}

export function slugifyHeading(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/`/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-');
}

export function extractMarkdownSections(markdown: string): MarkdownSection[] {
  const sections: MarkdownSection[] = [];
  const lines = markdown.split('\n');
  let inFence = false;

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const match = /^(##|###)\s+(.+)$/.exec(line.trim());
    if (!match) continue;
    const level = match[1] === '##' ? 2 : 3;
    const title = match[2].trim();
    const id = slugifyHeading(title);
    if (!id) continue;
    sections.push({ id, level, title });
  }

  return sections;
}

export function scrollToSection(id: string): void {
  const target = document.getElementById(id);
  if (!target) return;
  target.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

export function resolveMarkdownHref(href: string, sourcePath: string): string {
  if (
    href.startsWith('http://') ||
    href.startsWith('https://') ||
    href.startsWith('mailto:') ||
    href.startsWith('tel:')
  ) {
    return href;
  }

  if (href.startsWith('#')) {
    return href;
  }

  const [rawPath, hash = ''] = href.split('#', 2);
  const sourceDir = sourcePath.includes('/') ? sourcePath.slice(0, sourcePath.lastIndexOf('/')) : '';
  const combined = rawPath.startsWith('/')
    ? rawPath.slice(1)
    : [sourceDir, rawPath].filter(Boolean).join('/');
  const resolvedPath = normalizeRepoPath(combined);

  if (DOC_ROUTES[resolvedPath]) {
    return hash ? `${DOC_ROUTES[resolvedPath]}?section=${encodeURIComponent(hash)}` : DOC_ROUTES[resolvedPath];
  }

  return `${GITHUB_BLOB_ROOT}${resolvedPath}${hash ? `#${hash}` : ''}`;
}

export function MarkdownDocument({ markdown, sourcePath }: MarkdownDocumentProps) {
  return (
    <article className="markdown-shell rounded-[28px] border border-border bg-card p-7 shadow-[0_24px_80px_rgba(27,25,20,0.08)] md:p-10">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="mt-0 mb-6 scroll-mt-28 text-4xl font-semibold tracking-tight md:text-5xl">{children}</h1>,
          h2: ({ children }) => {
            const id = slugifyHeading(flattenText(children));
            return (
              <h2
                id={id || undefined}
                className="mt-14 mb-5 scroll-mt-28 border-t border-border pt-8 text-2xl font-semibold tracking-tight md:text-3xl"
              >
                {children}
              </h2>
            );
          },
          h3: ({ children }) => {
            const id = slugifyHeading(flattenText(children));
            return (
              <h3 id={id || undefined} className="mt-10 mb-4 scroll-mt-28 text-xl font-semibold tracking-tight">
                {children}
              </h3>
            );
          },
          h4: ({ children }) => {
            const id = slugifyHeading(flattenText(children));
            return (
              <h4 id={id || undefined} className="mt-8 mb-3 scroll-mt-28 text-lg font-semibold">
                {children}
              </h4>
            );
          },
          p: ({ children }) => <p className="my-5 leading-8 text-muted-foreground">{children}</p>,
          ul: ({ children }) => <ul className="my-5 list-disc space-y-3 pl-6 text-muted-foreground">{children}</ul>,
          ol: ({ children }) => <ol className="my-5 list-decimal space-y-3 pl-6 text-muted-foreground">{children}</ol>,
          li: ({ children }) => <li className="leading-8">{children}</li>,
          a: ({ href, children }) => {
            const resolved = href ? resolveMarkdownHref(href, sourcePath) : undefined;
            const isSectionAnchor = resolved?.startsWith('#') && !resolved.startsWith('#/');
            return (
              <a
                href={resolved}
                onClick={
                  isSectionAnchor
                    ? (event) => {
                        event.preventDefault();
                        scrollToSection(resolved.slice(1));
                      }
                    : undefined
                }
                className="font-medium text-foreground underline decoration-border underline-offset-4 transition hover:decoration-foreground"
              >
                {children}
              </a>
            );
          },
          strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
          code: ({ className, children }) => {
            const block = Boolean(className);
            if (block) {
              return <code className={className}>{children}</code>;
            }
            return <code className="rounded bg-muted px-1.5 py-0.5 text-[0.95em] text-foreground">{children}</code>;
          },
          pre: ({ children }) => (
            <pre className="my-6 overflow-x-auto rounded-[20px] border border-border bg-accent p-5 text-sm leading-7 text-foreground">
              {children}
            </pre>
          ),
          hr: () => <hr className="my-10 border-border" />,
          blockquote: ({ children }) => (
            <blockquote className="my-6 border-l-2 border-border pl-5 text-muted-foreground">{children}</blockquote>
          ),
          table: ({ children }) => (
            <div className="my-8 overflow-x-auto">
              <table className="min-w-full border-collapse text-left text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="border-b border-border bg-muted">{children}</thead>,
          tbody: ({ children }) => <tbody>{children}</tbody>,
          tr: ({ children }) => <tr className="border-b border-border last:border-b-0">{children}</tr>,
          th: ({ children }) => <th className="px-4 py-3 font-semibold text-foreground">{children}</th>,
          td: ({ children }) => <td className="px-4 py-3 align-top text-muted-foreground">{children}</td>
        }}
      >
        {markdown}
      </ReactMarkdown>
    </article>
  );
}
