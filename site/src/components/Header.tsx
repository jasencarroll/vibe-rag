import { Link, NavLink } from 'react-router';

const linkBase =
  'rounded-md px-3 py-2 text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground';

const navClass = ({ isActive }: { isActive: boolean }) =>
  isActive ? `${linkBase} bg-muted text-foreground` : linkBase;

export function Header() {
  return (
    <header className="border-b border-border bg-card/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center gap-3 no-underline">
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-border bg-accent text-sm">
            vr
          </span>
          <div>
            <div className="text-lg font-semibold tracking-tight text-foreground">vibe-rag</div>
            <div className="text-xs uppercase tracking-[0.22em] text-muted-foreground">
              repo memory for agents
            </div>
          </div>
        </Link>
        <nav className="flex items-center gap-1">
          <NavLink to="/readme" className={navClass}>
            README
          </NavLink>
          <NavLink to="/setup" className={navClass}>
            Setup
          </NavLink>
          <NavLink to="/guide" className={navClass}>
            User Guide
          </NavLink>
          <a
            href="https://github.com/jasencarroll/vibe-rag"
            className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground no-underline transition hover:opacity-90"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}

