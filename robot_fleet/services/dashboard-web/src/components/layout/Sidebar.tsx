import { NavLink } from 'react-router-dom'
import {
  Home,
  Bot,
  Target,
  GitBranch,
  Globe,
  FileText,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react'
import { cn } from '../../lib/utils'

const navigation = [
  { name: 'Home', href: '/', icon: Home },
  { name: 'Robots', href: '/robots', icon: Bot },
  { name: 'Goals', href: '/goals', icon: Target },
  { name: 'Plans', href: '/plans', icon: GitBranch },
  { name: 'World State', href: '/world', icon: Globe },
  { name: 'Planners', href: '/planners', icon: FileText },
  { name: 'Allocators', href: '/allocators', icon: Network },
]

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={cn(
        'fixed left-0 top-0 h-full bg-surface border-r border-border-subtle z-40 flex flex-col transition-[width] duration-200 ease-out',
        collapsed ? 'w-14' : 'w-60'
      )}
    >
      {/* Logo */}
      <div className="h-14 flex items-center border-b border-border-subtle shrink-0 px-3">
        {collapsed ? (
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyber-500 to-violet-500 flex items-center justify-center shadow-glow-sm mx-auto">
            <Bot className="w-4 h-4 text-white" />
          </div>
        ) : (
          <div className="flex items-center gap-2.5 px-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyber-500 to-violet-500 flex items-center justify-center shadow-glow-sm shrink-0">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <div className="overflow-hidden">
              <h1 className="text-sm font-semibold text-white leading-tight whitespace-nowrap">Robot Fleet</h1>
              <p className="text-2xs text-[var(--color-text-muted)] font-mono uppercase tracking-widest whitespace-nowrap">
                Mission Control
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto overflow-x-hidden">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            title={collapsed ? item.name : undefined}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-2.5 rounded-lg text-[13px] font-medium transition-colors duration-150',
                collapsed ? 'justify-center px-0 py-2' : 'px-3 py-2',
                isActive
                  ? 'bg-cyber-500/10 text-cyber-400 shadow-[inset_0_0_0_1px_rgba(6,182,212,0.2)]'
                  : 'text-[var(--color-text-secondary)] hover:text-white hover:bg-[var(--color-surface-raised)]'
              )
            }
          >
            <item.icon className="w-[18px] h-[18px] shrink-0" />
            {!collapsed && <span className="truncate">{item.name}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle + status */}
      <div className="px-2 pb-3 pt-2 border-t border-border-subtle shrink-0 space-y-2">
        {!collapsed && (
          <div className="bg-[var(--color-surface-raised)] rounded-lg p-3 mx-1">
            <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>System Online</span>
            </div>
            <p className="text-2xs text-[var(--color-text-muted)] font-mono mt-1.5">
              Fleet Manager: localhost:50051
            </p>
          </div>
        )}
        <button
          onClick={onToggle}
          className={cn(
            'flex items-center gap-2 rounded-lg text-[13px] text-[var(--color-text-muted)] hover:text-white hover:bg-[var(--color-surface-raised)] transition-colors duration-150 w-full',
            collapsed ? 'justify-center px-0 py-2' : 'px-4 py-2'
          )}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed
            ? <PanelLeftOpen className="w-[18px] h-[18px]" />
            : <>
                <PanelLeftClose className="w-[18px] h-[18px]" />
                <span>Collapse</span>
              </>
          }
        </button>
      </div>
    </aside>
  )
}
