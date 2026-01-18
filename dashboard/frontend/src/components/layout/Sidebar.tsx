import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Bot,
  Target,
  GitBranch,
  Play,
  Globe,
  Settings,
  FileText,
  Network
} from 'lucide-react'
import { cn } from '../../lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Robots', href: '/robots', icon: Bot },
  { name: 'Goals', href: '/goals', icon: Target },
  { name: 'Plans', href: '/plans', icon: GitBranch },
  { name: 'World State', href: '/world', icon: Globe },
  { name: 'Planners', href: '/planners', icon: FileText },
  { name: 'Allocators', href: '/allocators', icon: Network },
]

export function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-full w-64 bg-slate-900/80 backdrop-blur-sm border-r border-slate-800 z-40">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyber-500 to-violet-500 flex items-center justify-center">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-white">Robot Fleet</h1>
            <p className="text-xs text-slate-500 font-mono">MISSION CONTROL</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="p-4 space-y-1">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                isActive
                  ? 'bg-cyber-500/10 text-cyber-400 border border-cyber-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              )
            }
          >
            <item.icon className="w-5 h-5" />
            {item.name}
          </NavLink>
        ))}
      </nav>

      {/* Quick Actions */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-800">
        <div className="bg-slate-800/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>System Online</span>
          </div>
          <p className="text-xs text-slate-500 font-mono">
            Fleet Manager: localhost:50051
          </p>
        </div>
      </div>
    </aside>
  )
}
