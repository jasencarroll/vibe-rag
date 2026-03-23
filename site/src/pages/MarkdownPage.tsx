import { MarkdownDocument } from '../components/MarkdownDocument';

type MarkdownPageProps = {
  title: string;
  kicker: string;
  sourcePath: string;
  markdown: string;
};

export function MarkdownPage({ title, kicker, sourcePath, markdown }: MarkdownPageProps) {
  return (
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
  );
}
