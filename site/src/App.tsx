import { useEffect } from 'react';
import { Route, Routes, useLocation } from 'react-router';
import { Header } from './components/Header';
import { HomePage } from './pages/HomePage';
import { MarkdownPage } from './pages/MarkdownPage';
import { docs } from './site-docs';

function ScrollRestoration() {
  const location = useLocation();

  useEffect(() => {
    const search = new URLSearchParams(location.search);
    if (search.get('section')) {
      return;
    }
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [location.pathname, location.search]);

  return null;
}

export function App() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <ScrollRestoration />
      <Header />
      <main className="mx-auto max-w-6xl px-6 pb-24 pt-14">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route
            path="/readme"
            element={
              <MarkdownPage
                title={docs.readme.title}
                kicker={docs.readme.kicker}
                sourcePath={docs.readme.sourcePath}
                markdown={docs.readme.markdown}
              />
            }
          />
          <Route
            path="/setup"
            element={
              <MarkdownPage
                title={docs.setup.title}
                kicker={docs.setup.kicker}
                sourcePath={docs.setup.sourcePath}
                markdown={docs.setup.markdown}
              />
            }
          />
          <Route
            path="/maintainer"
            element={
              <MarkdownPage
                title={docs.maintainer.title}
                kicker={docs.maintainer.kicker}
                sourcePath={docs.maintainer.sourcePath}
                markdown={docs.maintainer.markdown}
              />
            }
          />
          <Route
            path="/guide"
            element={
              <MarkdownPage
                title={docs.guide.title}
                kicker={docs.guide.kicker}
                sourcePath={docs.guide.sourcePath}
                markdown={docs.guide.markdown}
              />
            }
          />
        </Routes>
      </main>
    </div>
  );
}
