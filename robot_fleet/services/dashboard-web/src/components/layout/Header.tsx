import { useLocation } from 'react-router-dom'

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/robots': 'Robot Management',
  '/goals': 'Goals',
  '/plans': 'Plan Builder',
  '/world': 'World State',
  '/planners': 'Planners',
  '/allocators': 'Allocators',
}

export function Header() {
  const location = useLocation()
  const title = pageTitles[location.pathname] || 'Dashboard'

  return (
    <header className="h-14 bg-surface/80 backdrop-blur-md border-b border-border-subtle flex items-center justify-between px-8 sticky top-0 z-30">
      <h2 className="text-sm font-semibold text-white tracking-tight">{title}</h2>
      <p className="text-xs text-[var(--color-text-muted)] font-mono tabular-nums">
        {new Date().toLocaleDateString('en-US', {
          weekday: 'short',
          month: 'short',
          day: 'numeric',
          year: 'numeric',
        })}
      </p>
    </header>
  )
}
