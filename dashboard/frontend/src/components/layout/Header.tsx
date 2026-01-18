import { useLocation } from 'react-router-dom'

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/robots': 'Robot Management',
  '/goals': 'Goals',
  '/plans': 'Plan Builder',
  '/world': 'World State',
}

export function Header() {
  const location = useLocation()

  const title = pageTitles[location.pathname] || 'Dashboard'

  return (
    <header className="h-16 bg-slate-900/50 backdrop-blur-sm border-b border-slate-800 flex items-center px-6 sticky top-0 z-30">
      <div>
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        <p className="text-xs text-slate-500 font-mono">
          {new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
          })}
        </p>
      </div>
    </header>
  )
}
