import { ReactNode } from 'react'
import { cn } from '../../lib/utils'

interface EmptyStateProps {
  icon: ReactNode
  title: string
  description: string
  action?: ReactNode
  className?: string
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center py-12 text-center', className)}>
      <div className="w-14 h-14 rounded-xl bg-surface-overlay flex items-center justify-center mb-4 text-[var(--color-text-muted)]">
        {icon}
      </div>
      <h3 className="text-base font-semibold text-white mb-1.5">{title}</h3>
      <p className="text-sm text-[var(--color-text-secondary)] max-w-sm mb-5">{description}</p>
      {action}
    </div>
  )
}
