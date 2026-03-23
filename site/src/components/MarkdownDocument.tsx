import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type MarkdownDocumentProps = {
  markdown: string;
  sourcePath: string;
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

export function resolveMarkdownHref(href: string, sourcePath: string): string {
  if (
    href.startsWith('http://') ||
    href.startsWith('https://') ||
    href.startsWith('mailto:') ||
    href.startsWith('tel:') ||
    href.startsWith('#')
  ) {
    return href;
  }

  const [rawPath, hash = ''] = href.split('#', 2);
  const sourceDir = sourcePath.includes('/') ? sourcePath.slice(0, sourcePath.lastIndexOf('/')) : '';
  const combined = rawPath.startsWith('/')
    ? rawPath.slice(1)
    : [sourceDir, rawPath].filter(Boolean).join('/');
  const resolvedPath = normalizeRepoPath(combined);

  if (DOC_ROUTES[resolvedPath]) {
    return DOC_ROUTES[resolvedPath];
  }

  return `${GITHUB_BLOB_ROOT}${resolvedPath}${hash ? `#${hash}` : ''}`;
}

export function MarkdownDocument({ markdown, sourcePath }: MarkdownDocumentProps) {
  return (
    <article className="markdown-shell rounded-[28px] border border-border bg-card p-7 shadow-[0_24px_80px_rgba(27,25,20,0.08)] md:p-10">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="mt-0 mb-6 text-4xl font-semibold tracking-tight md:text-5xl">{children}</h1>,
          h2: ({ children }) => (
            <h2 className="mt-14 mb-5 border-t border-border pt-8 text-2xl font-semibold tracking-tight md:text-3xl">
              {children}
            </h2>
          ),
          h3: ({ children }) => <h3 className="mt-10 mb-4 text-xl font-semibold tracking-tight">{children}</h3>,
          h4: ({ children }) => <h4 className="mt-8 mb-3 text-lg font-semibold">{children}</h4>,
          p: ({ children }) => <p className="my-5 leading-8 text-muted-foreground">{children}</p>,
          ul: ({ children }) => <ul className="my-5 list-disc space-y-3 pl-6 text-muted-foreground">{children}</ul>,
          ol: ({ children }) => <ol className="my-5 list-decimal space-y-3 pl-6 text-muted-foreground">{children}</ol>,
          li: ({ children }) => <li className="leading-8">{children}</li>,
          a: ({ href, children }) => (
            <a
              href={href ? resolveMarkdownHref(href, sourcePath) : undefined}
              className="font-medium text-foreground underline decoration-border underline-offset-4 transition hover:decoration-foreground"
            >
              {children}
            </a>
          ),
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
