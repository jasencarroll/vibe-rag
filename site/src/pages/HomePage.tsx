import { Link } from 'react-router';
import { MarkdownDocument } from '../components/MarkdownDocument';
import { docs } from '../site-docs';

const supportedClients = ['Vibe', 'Codex', 'Claude Code', 'Gemini CLI'];

const highlights = [
  {
    title: 'Session-start briefing',
    body:
      'Front-load repo pulse, dirt, stale hazards, live decisions, and task-relevant retrieval before the first real turn.'
  },
  {
    title: 'Semantic search',
    body:
      'Search code and docs through one MCP server instead of rebuilding the same retrieval layer in every client.'
  },
  {
    title: 'Durable memory',
    body:
      'Keep project decisions, constraints, and maintainer facts accessible across sessions without introducing external infrastructure.'
  }
];

const entryPoints = [
  {
    title: 'README',
    body: 'Product identity, quick start, support levels, and client scaffolding.',
    href: '/readme'
  },
  {
    title: 'Setup Guide',
    body: 'Install path, provider behavior, trust, and smoke-test flow.',
    href: '/setup'
  },
  {
    title: 'User Guide',
    body: 'Normal operator flow, memory hygiene, and retrieval order.',
    href: '/guide'
  }
];

export function HomePage() {
  return (
    <div className="space-y-20">
      <section className="grid gap-10 border-b border-border pb-18 lg:grid-cols-[1.08fr_0.92fr]">
        <div className="space-y-7">
          <div className="flex flex-wrap gap-2">
            {supportedClients.map((client) => (
              <span
                key={client}
                className="rounded-full border border-border bg-card px-3 py-1 text-xs uppercase tracking-[0.18em] text-muted-foreground"
              >
                {client}
              </span>
            ))}
          </div>
          <div className="space-y-5">
            <h1 className="max-w-4xl text-5xl leading-tight font-semibold tracking-tight md:text-7xl">
              Semantic repo search and memory for coding agents.
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-muted-foreground">
              The site is intentionally primitive: the repo docs are the source of truth, and the SPA
              is only a cleaner shell for reading them. Packaging quality and session-start quality
              matter more than marketing copy.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link
              to="/setup"
              className="rounded-md bg-primary px-5 py-3 text-sm font-medium text-primary-foreground no-underline transition hover:opacity-90"
            >
              Start with setup
            </Link>
            <Link
              to="/guide"
              className="rounded-md border border-border bg-card px-5 py-3 text-sm font-medium text-foreground no-underline transition hover:bg-muted"
            >
              Open the guide
            </Link>
          </div>
        </div>
        <div className="rounded-[28px] border border-border bg-card p-6 shadow-[0_24px_80px_rgba(27,25,20,0.08)]">
          <div className="mb-5 flex items-center justify-between border-b border-border pb-4">
            <div>
              <div className="text-sm uppercase tracking-[0.18em] text-muted-foreground">briefing</div>
              <div className="mt-1 text-lg text-foreground">startup context shape</div>
            </div>
            <span className="rounded-full bg-accent px-3 py-1 text-xs text-muted-foreground">SessionStart</span>
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap text-sm leading-7 text-foreground">
{`vibe-rag | repo-id | main | 2 modified files

You were last here 9 hours ago.

! Index may be stale
! 2 files modified but not committed

Decisions:
- Session bootstrap path and config generation

Code: src/vibe_rag/cli.py
Docs: docs/setup-guide.md`}
          </pre>
        </div>
      </section>

      <section className="space-y-8">
        <div className="max-w-2xl">
          <div className="mb-3 text-xs uppercase tracking-[0.22em] text-muted-foreground">core loop</div>
          <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
            The product is a briefing layer, a retrieval layer, and a memory layer.
          </h2>
        </div>
        <div className="grid gap-5 md:grid-cols-3">
          {highlights.map((item) => (
            <article key={item.title} className="rounded-[24px] border border-border bg-card p-6">
              <div className="mb-4 text-xs uppercase tracking-[0.18em] text-muted-foreground">layer</div>
              <h3 className="mb-4 text-xl font-semibold">{item.title}</h3>
              <p className="leading-7 text-muted-foreground">{item.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="space-y-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="mb-3 text-xs uppercase tracking-[0.22em] text-muted-foreground">canonical docs</div>
            <h2 className="text-3xl font-semibold tracking-tight">These pages are rendered from the repo markdown.</h2>
          </div>
          <a
            href="https://github.com/jasencarroll/vibe-rag"
            className="text-sm text-muted-foreground no-underline transition hover:text-foreground"
          >
            Open the repo
          </a>
        </div>
        <div className="grid gap-5 md:grid-cols-3">
          {entryPoints.map((item) => (
            <Link
              key={item.title}
              to={item.href}
              className="flex h-full flex-col justify-between rounded-[24px] border border-border bg-card p-6 no-underline transition hover:-translate-y-0.5 hover:bg-muted"
            >
              <div>
                <div className="mb-3 text-xs uppercase tracking-[0.18em] text-muted-foreground">doc</div>
                <h3 className="mb-4 text-xl font-semibold text-foreground">{item.title}</h3>
                <p className="leading-7 text-muted-foreground">{item.body}</p>
              </div>
              <div className="mt-8 text-sm text-foreground">Open</div>
            </Link>
          ))}
        </div>
      </section>

      <section className="space-y-6">
        <div className="max-w-2xl">
          <div className="mb-3 text-xs uppercase tracking-[0.22em] text-muted-foreground">overview</div>
          <h2 className="text-3xl font-semibold tracking-tight">README, rendered directly.</h2>
        </div>
        <MarkdownDocument markdown={docs.readme.markdown} sourcePath={docs.readme.sourcePath} />
      </section>
    </div>
  );
}
