import { Route, Routes } from 'react-router';
import { Header } from './components/Header';
import { HomePage } from './pages/HomePage';
import { MarkdownPage } from './pages/MarkdownPage';
import { docs } from './site-docs';

export function App() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />
      <main className="mx-auto max-w-6xl px-6 pb-24 pt-14">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route
            path="/readme"
            element={<MarkdownPage title={docs.readme.title} kicker={docs.readme.kicker} markdown={docs.readme.markdown} />}
          />
          <Route
            path="/setup"
            element={<MarkdownPage title={docs.setup.title} kicker={docs.setup.kicker} markdown={docs.setup.markdown} />}
          />
          <Route
            path="/guide"
            element={<MarkdownPage title={docs.guide.title} kicker={docs.guide.kicker} markdown={docs.guide.markdown} />}
          />
        </Routes>
      </main>
    </div>
  );
}

