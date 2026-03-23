import { describe, expect, it } from 'vitest';
import { resolveMarkdownHref } from './MarkdownDocument';

describe('resolveMarkdownHref', () => {
  it('routes repo docs into the SPA', () => {
    expect(resolveMarkdownHref('setup-guide.md', 'docs/user-guide.md')).toBe('#/setup');
    expect(resolveMarkdownHref('docs/user-guide.md', 'README.md')).toBe('#/guide');
    expect(resolveMarkdownHref('docs/maintainer-guide.md', 'README.md')).toBe('#/maintainer');
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
