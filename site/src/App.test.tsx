import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router';
import { App } from './App';

function renderRoute(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  Object.defineProperty(window, 'scrollTo', {
    value: vi.fn(),
    writable: true,
  });
  Object.defineProperty(Element.prototype, 'scrollIntoView', {
    value: vi.fn(),
    writable: true,
  });
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    callback(0);
    return 0;
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('site routes', () => {
  it('renders the home page', () => {
    renderRoute('/');
    expect(screen.getByRole('heading', { name: /semantic repo search and memory for coding agents/i })).toBeTruthy();
  });

  it('renders all canonical doc pages', () => {
    const cases = [
      ['/readme', 'README'],
      ['/setup', 'Setup Guide'],
      ['/guide', 'User Guide'],
      ['/maintainer', 'Maintainer Guide'],
    ] as const;

    for (const [route, heading] of cases) {
      const view = renderRoute(route);
      expect(screen.getByRole('heading', { name: heading, level: 1 })).toBeTruthy();
      view.unmount();
    }
  });
});

describe('markdown page behavior', () => {
  it('renders a sidebar with section links', () => {
    renderRoute('/setup');
    expect(screen.getByRole('button', { name: '1. Install the Tools' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '2. Scaffold a Repo' })).toBeTruthy();
  });

  it('restores top scroll on normal route changes', () => {
    renderRoute('/');
    vi.mocked(window.scrollTo).mockClear();

    fireEvent.click(screen.getByRole('link', { name: 'Start with setup' }));

    expect(screen.getByRole('heading', { name: 'Setup Guide', level: 1 })).toBeTruthy();
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, left: 0, behavior: 'auto' });
  });

  it('preserves section scrolling when a section query is present', () => {
    renderRoute('/setup?section=1-install-the-tools');

    expect(window.scrollTo).not.toHaveBeenCalled();
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });
});
