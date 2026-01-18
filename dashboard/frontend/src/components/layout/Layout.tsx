import { ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { Header } from './Header'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-slate-950 grid-bg">
      {/* Radial glow effect at top */}
      <div className="fixed inset-0 radial-glow pointer-events-none" />
      
      <div className="flex relative">
        <Sidebar />
        <div className="flex-1 ml-64">
          <Header />
          <main className="p-6 min-h-[calc(100vh-64px)]">
            {children}
          </main>
        </div>
      </div>
    </div>
  )
}
