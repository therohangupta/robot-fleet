import {
  CheckCircle,
  Clock,
  AlertCircle,
  Circle,
  Video,
  Activity,
  Terminal,
  ChevronDown,
  ChevronRight,
  MessageSquare,
} from 'lucide-react'
import { useState } from 'react'
import { Modal } from '../common/Modal'
import { StatusBadge } from '../common/StatusBadge'
import { TaskIdChip } from './TaskIdChip'
import { cn } from '../../lib/utils'
import type { Task } from '../../types'

interface RobotExecutionModalProps {
  isOpen: boolean
  onClose: () => void
  robotId: string
  robotType: string
  tasks: Task[]
  allTasks: Task[]
  elapsedTimes: Map<number, number>
}

const statusIcon: Record<string, React.ReactNode> = {
  completed: <CheckCircle className="w-4 h-4 text-emerald-400" />,
  in_progress: <Clock className="w-4 h-4 text-amber-400 animate-pulse" />,
  failed: <AlertCircle className="w-4 h-4 text-red-400" />,
  pending: <Circle className="w-4 h-4 text-[var(--color-text-secondary)]" />,
}

function formatElapsed(ms: number): string {
  const secs = Math.floor(ms / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const remainder = secs % 60
  return `${mins}m ${remainder}s`
}

export function RobotExecutionModal({
  isOpen,
  onClose,
  robotId,
  robotType,
  tasks,
  allTasks,
  elapsedTimes,
}: RobotExecutionModalProps) {
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set())

  const taskMap = new Map(allTasks.map(t => [t.task_id, t]))

  const currentTask = tasks.find(t => t.status === 'in_progress')
  const completedTasks = tasks.filter(t => t.status === 'completed' || t.status === 'failed')
  const pendingTasks = tasks.filter(t => t.status === 'pending')

  const toggleResult = (taskId: number) => {
    const next = new Set(expandedResults)
    if (next.has(taskId)) next.delete(taskId)
    else next.add(taskId)
    setExpandedResults(next)
  }

  const depsReady = (task: Task) =>
    task.dependency_task_ids.every(depId => {
      const dep = taskMap.get(depId)
      return dep && dep.status === 'completed'
    })

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`@${robotId} (${robotType})`} size="wide">
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_0.8fr] gap-6 max-h-[75vh] overflow-y-auto pr-1">
        {/* Left column: task info */}
        <div className="space-y-5">
          {/* Current task */}
          <div>
            <h4 className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-3">
              Current Task
            </h4>
            {currentTask ? (
              <div className="p-4 rounded-lg border-2 border-amber-500/40 bg-amber-500/5">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-amber-400 animate-pulse" />
                  <span className="text-xs font-mono text-amber-400">#{currentTask.task_id}</span>
                  <StatusBadge status="in_progress" />
                  {elapsedTimes.has(currentTask.task_id) && (
                    <span className="text-xs font-mono text-amber-300 ml-auto">
                      {formatElapsed(elapsedTimes.get(currentTask.task_id)!)}
                    </span>
                  )}
                </div>
                <p className="text-white text-sm leading-relaxed">{currentTask.description}</p>
                {currentTask.dependency_task_ids.length > 0 && (
                  <div className="flex items-center gap-1.5 mt-2 text-xs text-[var(--color-text-secondary)]">
                    <span>Deps:</span>
                    {currentTask.dependency_task_ids.map(depId => (
                      <TaskIdChip key={depId} taskId={depId} task={taskMap.get(depId)} />
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4 rounded-lg border border-border bg-surface-overlay/40 text-center text-sm text-[var(--color-text-muted)]">
                {completedTasks.length === tasks.length ? 'All tasks complete' : 'Idle — waiting for dependencies'}
              </div>
            )}
          </div>

          {/* Pending tasks */}
          {pendingTasks.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-3">
                Queued ({pendingTasks.length})
              </h4>
              <div className="space-y-2">
                {pendingTasks.map(task => (
                  <div
                    key={task.task_id}
                    className="flex items-center gap-3 p-3 rounded-lg border border-border-subtle bg-surface-overlay/30"
                  >
                    {statusIcon.pending}
                    <span className="text-xs font-mono text-[var(--color-text-muted)]">#{task.task_id}</span>
                    <p className="text-sm text-[var(--color-text-secondary)] flex-1 truncate">{task.description}</p>
                    <span className={cn(
                      'text-xs px-1.5 py-0.5 rounded',
                      depsReady(task)
                        ? 'bg-emerald-500/10 text-emerald-400'
                        : 'bg-surface-elevated/50 text-[var(--color-text-muted)]'
                    )}>
                      {depsReady(task) ? 'Ready' : 'Blocked'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Completed tasks */}
          {completedTasks.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-3">
                Completed ({completedTasks.length})
              </h4>
              <div className="space-y-2">
                {[...completedTasks].reverse().map(task => (
                  <div
                    key={task.task_id}
                    className={cn(
                      'rounded-lg border bg-surface-overlay/30',
                      task.status === 'failed' ? 'border-red-500/25' : 'border-border-subtle'
                    )}
                  >
                    <div
                      className="flex items-center gap-3 p-3 cursor-pointer"
                      onClick={() => task.result && toggleResult(task.task_id)}
                    >
                      {statusIcon[task.status] || statusIcon.pending}
                      <span className={cn(
                        'text-xs font-mono',
                        task.status === 'failed' ? 'text-red-400' : 'text-emerald-400'
                      )}>
                        #{task.task_id}
                      </span>
                      <p className="text-sm text-[var(--color-text-secondary)] flex-1 truncate">{task.description}</p>
                      <StatusBadge status={task.status} />
                      {task.result && (
                        expandedResults.has(task.task_id)
                          ? <ChevronDown className="w-3.5 h-3.5 text-[var(--color-text-secondary)]" />
                          : <ChevronRight className="w-3.5 h-3.5 text-[var(--color-text-secondary)]" />
                      )}
                    </div>
                    {expandedResults.has(task.task_id) && task.result && (
                      <div className="px-3 pb-3">
                        <div className="bg-surface/80 rounded p-3 border border-border">
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <MessageSquare className="w-3.5 h-3.5 text-[var(--color-text-muted)]" />
                            <span className="text-xs font-medium text-[var(--color-text-muted)]">Result</span>
                          </div>
                          <pre className="text-xs text-[var(--color-text-secondary)] whitespace-pre-wrap font-mono leading-relaxed">
                            {task.result}
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right column: telemetry placeholders */}
        <div className="space-y-4">
          {/* Video feed placeholder */}
          <div className="rounded-lg border border-border bg-surface-overlay/60 overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border-subtle">
              <Video className="w-4 h-4 text-[var(--color-text-muted)]" />
              <span className="text-sm font-medium text-[var(--color-text-secondary)]">Live Video Feed</span>
            </div>
            <div className="aspect-video flex flex-col items-center justify-center text-[var(--color-text-muted)] bg-surface/50">
              <Video className="w-10 h-10 mb-3 opacity-40" />
              <p className="text-sm">Video feed not connected</p>
              <p className="text-xs mt-1 font-mono opacity-50">
                /robots/{robotId}/video
              </p>
            </div>
          </div>

          {/* Joint states placeholder */}
          <div className="rounded-lg border border-border bg-surface-overlay/60 overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border-subtle">
              <Activity className="w-4 h-4 text-[var(--color-text-muted)]" />
              <span className="text-sm font-medium text-[var(--color-text-secondary)]">Joint States</span>
            </div>
            <div className="p-4">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[var(--color-text-muted)] border-b border-border-subtle">
                    <th className="text-left pb-2 font-medium">Joint</th>
                    <th className="text-right pb-2 font-medium">Position</th>
                    <th className="text-right pb-2 font-medium">Velocity</th>
                    <th className="text-right pb-2 font-medium">Torque</th>
                  </tr>
                </thead>
                <tbody className="text-[var(--color-text-muted)]">
                  {['base', 'shoulder', 'elbow', 'wrist_1', 'wrist_2', 'gripper'].map(joint => (
                    <tr key={joint} className="border-b border-border-subtle">
                      <td className="py-1.5 font-mono">{joint}</td>
                      <td className="py-1.5 text-right">—</td>
                      <td className="py-1.5 text-right">—</td>
                      <td className="py-1.5 text-right">—</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-xs text-[var(--color-text-muted)] font-mono mt-3 text-center opacity-50">
                /robots/{robotId}/joints
              </p>
            </div>
          </div>

          {/* Joint commands log placeholder */}
          <div className="rounded-lg border border-border bg-surface-overlay/60 overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border-subtle">
              <Terminal className="w-4 h-4 text-[var(--color-text-muted)]" />
              <span className="text-sm font-medium text-[var(--color-text-secondary)]">Joint Commands</span>
            </div>
            <div className="h-32 flex flex-col items-center justify-center text-[var(--color-text-muted)] bg-surface/50">
              <Terminal className="w-8 h-8 mb-2 opacity-40" />
              <p className="text-xs">Command stream not connected</p>
              <p className="text-xs mt-1 font-mono opacity-50">
                /robots/{robotId}/commands
              </p>
            </div>
          </div>
        </div>
      </div>
    </Modal>
  )
}
