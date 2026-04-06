import { ReactNode, ButtonHTMLAttributes } from 'react'
import { cn } from '../../lib/utils'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
}

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  className,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all duration-150',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyber-500 focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-bg)]',
        variant === 'primary' && 'bg-cyber-600 text-white hover:bg-cyber-500 active:bg-cyber-700',
        variant === 'secondary' && 'bg-surface-overlay text-white border border-border hover:border-border-strong hover:bg-surface-elevated',
        variant === 'danger' && 'bg-red-500/10 text-red-400 border border-red-500/25 hover:bg-red-500/20',
        variant === 'ghost' && 'text-[var(--color-text-secondary)] hover:text-white hover:bg-surface-overlay',
        size === 'sm' && 'px-2.5 py-1.5 text-xs',
        size === 'md' && 'px-3.5 py-2 text-sm',
        size === 'lg' && 'px-5 py-2.5 text-sm',
        disabled && 'opacity-40 cursor-not-allowed pointer-events-none',
        className
      )}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  )
}
