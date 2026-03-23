import { describe, expect, it } from 'vitest';
import { extractMarkdownSections, resolveMarkdownHref } from './MarkdownDocument';

describe('resolveMarkdownHref', () => {
  it('routes repo docs into the SPA', () => {
    expect(resolveMarkdownHref('setup-guide.md', 'docs/user-guide.md')).toBe('#/setup');
    expect(resolveMarkdownHref('docs/user-guide.md', 'README.md')).toBe('#/guide');
    expect(resolveMarkdownHref('docs/maintainer-guide.md', 'README.md')).toBe('#/maintainer');
  });

  it('preserves section targets for internal doc routes', () => {
    expect(resolveMarkdownHref('setup-guide.md#smoke-test', 'docs/user-guide.md')).toBe(
      '#/setup?section=smoke-test',
    );
  });

  it('routes legacy AGENTS links to the public maintainer page', () => {
    expect(resolveMarkdownHref('AGENTS.md', 'README.md')).toBe('#/maintainer');
  });

  it('keeps external links unchanged', () => {
    expect(resolveMarkdownHref('https://example.com', 'README.md')).toBe('https://example.com');
    expect(resolveMarkdownHref('mailto:test@example.com', 'README.md')).toBe('mailto:test@example.com');
  });

  it('falls back to github blob urls for non-routed repo files', () => {
    expect(resolveMarkdownHref('src/vibe_rag/cli.py', 'README.md')).toBe(
      'https://github.com/jasencarroll/vibe-rag/blob/main/src/vibe_rag/cli.py',
    );
    expect(resolveMarkdownHref('../CHANGELOG.md', 'docs/setup-guide.md')).toBe(
      'https://github.com/jasencarroll/vibe-rag/blob/main/CHANGELOG.md',
    );
  });
});

describe('extractMarkdownSections', () => {
  it('extracts h2 and h3 sections while ignoring fenced code blocks', () => {
    const markdown = `# Title

## First section

### Nested section

\`\`\`md
## not-a-real-section
\`\`\`

## Final section`;

    expect(extractMarkdownSections(markdown)).toEqual([
      { id: 'first-section', level: 2, title: 'First section' },
      { id: 'nested-section', level: 3, title: 'Nested section' },
      { id: 'final-section', level: 2, title: 'Final section' },
    ]);
  });
});
