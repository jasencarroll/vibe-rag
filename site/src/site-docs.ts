import readme from '../../README.md?raw';
import setupGuide from '../../docs/setup-guide.md?raw';
import userGuide from '../../docs/user-guide.md?raw';

export const docs = {
  readme: {
    title: 'README',
    kicker: 'overview',
    sourcePath: 'README.md',
    markdown: readme
  },
  setup: {
    title: 'Setup Guide',
    kicker: 'setup',
    sourcePath: 'docs/setup-guide.md',
    markdown: setupGuide
  },
  guide: {
    title: 'User Guide',
    kicker: 'day to day',
    sourcePath: 'docs/user-guide.md',
    markdown: userGuide
  }
} as const;
