import readme from '../../README.md?raw';
import setupGuide from '../../docs/setup-guide.md?raw';
import userGuide from '../../docs/user-guide.md?raw';

export const docs = {
  readme: {
    title: 'README',
    kicker: 'overview',
    markdown: readme
  },
  setup: {
    title: 'Setup Guide',
    kicker: 'setup',
    markdown: setupGuide
  },
  guide: {
    title: 'User Guide',
    kicker: 'day to day',
    markdown: userGuide
  }
} as const;

