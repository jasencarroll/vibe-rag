import { useEffect, useMemo } from 'react';
import { useLocation } from 'react-router';
import { extractMarkdownSections, MarkdownDocument, scrollToSection } from '../components/MarkdownDocument';

type MarkdownPageProps = {
  title: string;
  kicker: string;
  sourcePath: string;
  markdown: string;
};

export function MarkdownPage({ title, kicker, sourcePath, markdown }: MarkdownPageProps) {
  const location = useLocation();
  const sections = useMemo(() => extractMarkdownSections(markdown), [markdown]);

  useEffect(() => {
    const search = new URLSearchParams(location.search);
    const section = search.get('section');
    if (!section) return;
    requestAnimationFrame(() => {
      scrollToSection(section);
    });
  }, [location.search]);

  return (
    <div className="grid gap-10 lg:grid-cols-[260px_minmax(0,1fr)]">
      <aside className="self-start lg:sticky lg:top-28">
        <div className="rounded-[24px] border border-border bg-card p-5">
          <div className="mb-4 text-xs uppercase tracking-[0.22em] text-muted-foreground">on this page</div>
          <div className="flex gap-2 overflow-x-auto pb-1 lg:block lg:space-y-1">
            {sections.map((section) => (
              <button
                key={section.id}
                type="button"
                onClick={() => scrollToSection(section.id)}
                className={`min-w-fit rounded-md px-3 py-2 text-left text-sm transition hover:bg-muted hover:text-foreground lg:flex lg:w-full ${
                  section.level === 3
                    ? 'text-muted-foreground lg:pl-6'
                    : 'font-medium text-foreground'
                }`}
              >
                {section.title}
              </button>
            ))}
          </div>
        </div>
      </aside>
      <div className="space-y-8">
        <section className="max-w-3xl space-y-5">
          <div className="text-xs uppercase tracking-[0.22em] text-muted-foreground">{kicker}</div>
          <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">{title}</h1>
          <p className="text-lg leading-8 text-muted-foreground">
            This page is rendered from the markdown tracked in the main <span className="text-foreground">vibe-rag</span>{' '}
            repo. Change the docs there and the site updates with the next Pages build.
          </p>
        </section>
        <MarkdownDocument markdown={markdown} sourcePath={sourcePath} />
      </div>
    </div>
  );
}
