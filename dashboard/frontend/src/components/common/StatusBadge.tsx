import { cn, getStatusBgColor } from '../../lib/utils'

interface StatusBadgeProps {
  status: string
  className?: string
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border',
        getStatusBgColor(status),
        className
      )}
    >
      <span
        className={cn(
          'w-1.5 h-1.5 rounded-full',
          status === 'completed' || status === 'running' ? 'bg-emerald-400' :
          status === 'in_progress' || status === 'deploying' ? 'bg-amber-400 animate-pulse' :
          status === 'pending' || status === 'registered' ? 'bg-blue-400' :
          status === 'failed' || status === 'error' ? 'bg-red-400' :
          'bg-gray-400'
        )}
      />
      {status.replace(/_/g, ' ')}
    </span>
  )
}
