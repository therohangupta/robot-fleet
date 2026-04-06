import { ReactNode, useState, useEffect } from 'react'
import { Sidebar } from './Sidebar'

interface LayoutProps {
  children: ReactNode
}

const SIDEBAR_KEY = 'rf-sidebar-collapsed'

export function Layout({ children }: LayoutProps) {
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(SIDEBAR_KEY) === '1' } catch { return false }
  })

  useEffect(() => {
    try { localStorage.setItem(SIDEBAR_KEY, collapsed ? '1' : '0') } catch {}
  }, [collapsed])

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <div className="flex relative">
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
        <div
          className="flex-1 transition-[margin] duration-200 ease-out"
          style={{ marginLeft: collapsed ? 56 : 240 }}
        >
          <main className="px-8 py-6 min-h-screen">
            {children}
          </main>
        </div>
      </div>
    </div>
  )
}
