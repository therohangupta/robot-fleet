import { useCallback, useMemo, useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  CheckCircle,
  Clock,
  Circle,
  Bot,
  ChevronDown,
  ChevronRight,
  Activity,
  ScrollText,
  Play,
  RotateCcw,
} from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { StatusBadge } from '../components/common/StatusBadge'
import { DAGVisualization } from '../components/common/DAGVisualization'
import { TaskIdChip } from '../components/execution/TaskIdChip'
import { RobotExecutionModal } from '../components/execution/RobotExecutionModal'
import { plansApi, tasksApi, robotsApi, methodsApi, useRealtimeUpdates } from '../lib/api'
import { cn, getPlanningStrategyName, getAllocationStrategyName, setMethodData } from '../lib/utils'
import { GatewayRealtimeClient } from '@robot-fleet/client-sdk'
import type { Task } from '../types'

// ---------------------------------------------------------------------------
// Event log types
// ---------------------------------------------------------------------------

interface LogEntry {
  ts: number
  message: string
  level: 'info' | 'success' | 'error'
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatElapsed(ms: number): string {
  const secs = Math.floor(ms / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const remainder = secs % 60
  return `${mins}m ${remainder}s`
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function Execution() {
  useRealtimeUpdates()
  const { planId } = useParams<{ planId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Tracks whether execution was started during this session (so WS connects immediately)
  const [executionStarted, setExecutionStarted] = useState(false)

  const startMutation = useMutation({
    mutationFn: () => plansApi.start(Number(planId)),
    onSuccess: () => {
      setExecutionStarted(true)
      queryClient.invalidateQueries({ queryKey: ['plan', planId] })
      queryClient.invalidateQueries({ queryKey: ['plans'] })
    },
  })

  const copyAndRetryMutation = useMutation({
    mutationFn: () => plansApi.copy(Number(planId), {
      name: `${plan?.name || `Plan #${planId}`} (Retry)`,
      description: plan?.description || 'Retry of failed plan',
    }),
    onSuccess: (newPlan: any) => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      navigate(`/plans/${newPlan.plan_id}/execute`)
    },
  })

  // UI state
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['fleet', 'queue'])
  )
  const [selectedRobot, setSelectedRobot] = useState<string | null>(null)
  const [eventLog, setEventLog] = useState<LogEntry[]>([])

  // Elapsed time tracking: taskId -> startTime (Date.now())
  const taskStartTimes = useRef<Map<number, number>>(new Map())
  const [elapsedTimes, setElapsedTimes] = useState<Map<number, number>>(new Map())

  // Wall-clock timer for the whole execution
  const executionStartRef = useRef<number | null>(null)
  const [wallElapsed, setWallElapsed] = useState(0)

  // WebSocket tasks state (from dedicated execution WS)
  const [wsTasks, setWsTasks] = useState<Task[] | null>(null)
  const prevTasksRef = useRef<Map<number, Task>>(new Map())

  // ---------------------------------------------------------------------------
  // Data fetching (initial load + fallback)
  // ---------------------------------------------------------------------------

  const { data: plan } = useQuery({
    queryKey: ['plan', planId],
    queryFn: () => plansApi.get(Number(planId)),
    enabled: !!planId,
    refetchInterval: (query) => {
      const p = query.state.data
      return p?.execution_status === 'executing' ? 3000 : false
    },
  })

  const isPreview = !executionStarted &&
    plan?.execution_status !== 'executing' &&
    plan?.execution_status !== 'completed' &&
    plan?.execution_status !== 'failed'

  const { data: fetchedTasks = [] } = useQuery({
    queryKey: ['tasks', planId],
    queryFn: () => tasksApi.list({ plan_id: Number(planId) }),
    enabled: !!planId,
  })

  const { data: robots = [] } = useQuery({
    queryKey: ['robots'],
    queryFn: () => robotsApi.list(),
  })

  const { data: planners = [] } = useQuery({
    queryKey: ['planners'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'planner')),
  })

  const { data: allocators = [] } = useQuery({
    queryKey: ['allocators'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'allocator')),
  })

  useEffect(() => {
    setMethodData(planners, allocators)
  }, [planners, allocators])

  // Authoritative task list: prefer WS data when available
  const tasks: Task[] = wsTasks ?? fetchedTasks

  // Task lookup map
  const taskMap = useMemo(() => new Map(tasks.map(t => [t.task_id, t])), [tasks])

  // ---------------------------------------------------------------------------
  // Dedicated execution WebSocket
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!planId || isPreview) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsBaseUrl = `${protocol}//${window.location.host}`
    const client = new GatewayRealtimeClient({ wsBaseUrl })

    const ws = client.connectPlanExecution(Number(planId), (msg) => {
      if (msg.type === 'tasks_update' && Array.isArray((msg as any).tasks)) {
        setWsTasks((msg as any).tasks as Task[])
      }
    })

    return () => {
      ws.close()
    }
  }, [planId, isPreview])

  // ---------------------------------------------------------------------------
  // Diff tasks on every update → build event log + track elapsed
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const prevMap = prevTasksRef.current
    const newEntries: LogEntry[] = []
    const now = Date.now()

    for (const task of tasks) {
      const prev = prevMap.get(task.task_id)
      if (!prev) continue

      if (prev.status !== task.status) {
        if (task.status === 'in_progress') {
          taskStartTimes.current.set(task.task_id, now)
          newEntries.push({
            ts: now,
            message: `Task #${task.task_id} started on ${task.robot_id ?? '?'}`,
            level: 'info',
          })
        } else if (task.status === 'completed') {
          taskStartTimes.current.delete(task.task_id)
          const resultSnippet = task.result
            ? `: ${task.result.length > 80 ? task.result.slice(0, 80) + '…' : task.result}`
            : ''
          newEntries.push({
            ts: now,
            message: `Task #${task.task_id} completed on ${task.robot_id ?? '?'}${resultSnippet}`,
            level: 'success',
          })
        } else if (task.status === 'failed') {
          taskStartTimes.current.delete(task.task_id)
          newEntries.push({
            ts: now,
            message: `Task #${task.task_id} failed on ${task.robot_id ?? '?'}${task.result ? ': ' + task.result.slice(0, 80) : ''}`,
            level: 'error',
          })
        }
      }
    }

    if (newEntries.length > 0) {
      setEventLog(prev => [...newEntries.reverse(), ...prev].slice(0, 200))
    }

    // Update prev snapshot
    prevTasksRef.current = new Map(tasks.map(t => [t.task_id, t]))
  }, [tasks])

  // Seed prev tasks ref on first load so the initial state doesn't generate log entries
  useEffect(() => {
    if (fetchedTasks.length > 0 && prevTasksRef.current.size === 0) {
      prevTasksRef.current = new Map(fetchedTasks.map(t => [t.task_id, t]))

      // Seed start times for tasks already in_progress
      const now = Date.now()
      for (const t of fetchedTasks) {
        if (t.status === 'in_progress') {
          taskStartTimes.current.set(t.task_id, now)
        }
      }
    }
  }, [fetchedTasks])

  // ---------------------------------------------------------------------------
  // 1-second timer to tick elapsed counters
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now()
      const next = new Map<number, number>()
      taskStartTimes.current.forEach((start, taskId) => {
        next.set(taskId, now - start)
      })
      setElapsedTimes(next)

      // Wall-clock
      if (executionStartRef.current) {
        setWallElapsed(now - executionStartRef.current)
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  // Detect execution start
  useEffect(() => {
    if (!executionStartRef.current && tasks.some(t => t.status === 'in_progress' || t.status === 'completed')) {
      executionStartRef.current = Date.now()
    }
  }, [tasks])

  // ---------------------------------------------------------------------------
  // Derived data
  // ---------------------------------------------------------------------------

  const isCancelled = useCallback((t: Task) =>
    t.status === 'failed' && (t.result?.startsWith('Cancelled:') || t.result?.startsWith('Skipped:')),
    []
  )

  const taskGroups = useMemo(() => ({
    executing: tasks.filter(t => t.status === 'in_progress'),
    pending: tasks.filter(t => t.status === 'pending'),
    completed: tasks.filter(t => t.status === 'completed'),
    failed: tasks.filter(t => t.status === 'failed' && !isCancelled(t)),
    cancelled: tasks.filter(t => isCancelled(t)),
  }), [tasks, isCancelled])

  const totalTasks = tasks.length
  const completedCount = taskGroups.completed.length
  const executingCount = taskGroups.executing.length
  const pendingCount = taskGroups.pending.length
  const failedCount = taskGroups.failed.length
  const cancelledCount = taskGroups.cancelled.length

  const overallStatus = useMemo(() => {
    if ((failedCount > 0 || cancelledCount > 0) && executingCount === 0 && pendingCount === 0) return 'failed'
    if (completedCount === totalTasks && totalTasks > 0) return 'completed'
    if (executingCount > 0) return 'in_progress'
    return 'pending'
  }, [totalTasks, completedCount, executingCount, pendingCount, failedCount, cancelledCount])

  // Group tasks by robot for the fleet table
  const robotRows = useMemo(() => {
    const byRobot = new Map<string, Task[]>()
    for (const task of tasks) {
      const rid = task.robot_id ?? '__unassigned__'
      if (!byRobot.has(rid)) byRobot.set(rid, [])
      byRobot.get(rid)!.push(task)
    }

    return Array.from(byRobot.entries())
      .filter(([rid]) => rid !== '__unassigned__')
      .map(([rid, rTasks]) => {
        const robot = robots.find(r => r.robot_id === rid)
        const current = rTasks.find(t => t.status === 'in_progress')
        const succeeded = rTasks.filter(t => t.status === 'completed').length
        const failed = rTasks.filter(t => t.status === 'failed' && !isCancelled(t)).length
        const cancelled = rTasks.filter(t => isCancelled(t)).length
        const total = rTasks.length
        const finished = succeeded + failed + cancelled === total
        const hasFailed = failed > 0 || cancelled > 0

        return {
          robotId: rid,
          robotType: robot?.robot_type ?? rTasks[0]?.robot_type ?? 'unknown',
          currentTask: current ?? null,
          succeededCount: succeeded,
          failedCount: failed,
          cancelledCount: cancelled,
          totalCount: total,
          finished,
          hasFailed,
          tasks: rTasks,
        }
      })
      .sort((a, b) => {
        const score = (r: typeof a) => r.currentTask ? 0 : r.finished ? (r.hasFailed ? 3 : 2) : 1
        return score(a) - score(b)
      })
  }, [tasks, robots])

  // Pending tasks for the queue table
  const pendingTasksSorted = useMemo(() => {
    return [...taskGroups.pending].sort((a, b) => {
      const aReady = a.dependency_task_ids.every(d => taskMap.get(d)?.status === 'completed')
      const bReady = b.dependency_task_ids.every(d => taskMap.get(d)?.status === 'completed')
      if (aReady && !bReady) return -1
      if (!aReady && bReady) return 1
      return a.task_id - b.task_id
    })
  }, [taskGroups.pending, taskMap])

  // Completed tasks for the completed table (most recent first)
  const completedTasksSorted = useMemo(() => {
    return [...taskGroups.completed, ...taskGroups.failed, ...taskGroups.cancelled].reverse()
  }, [taskGroups.completed, taskGroups.failed, taskGroups.cancelled])

  // ---------------------------------------------------------------------------
  // Section toggle
  // ---------------------------------------------------------------------------

  const toggleSection = useCallback((section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(section)) next.delete(section)
      else next.add(section)
      return next
    })
  }, [])

  // Auto-expand completed section when plan finishes
  useEffect(() => {
    if (overallStatus === 'completed' || overallStatus === 'failed') {
      setExpandedSections(prev => new Set([...prev, 'completed']))
    }
  }, [overallStatus])

  // ---------------------------------------------------------------------------
  // Helpers for dependency readiness
  // ---------------------------------------------------------------------------

  const depsReady = useCallback((task: Task) =>
    task.dependency_task_ids.every(d => taskMap.get(d)?.status === 'completed'),
    [taskMap]
  )

  // ---------------------------------------------------------------------------
  // Selected robot data for modal
  // ---------------------------------------------------------------------------

  const selectedRobotRow = selectedRobot
    ? robotRows.find(r => r.robotId === selectedRobot)
    : null

  // ---------------------------------------------------------------------------
  // Progress bar segment widths
  // ---------------------------------------------------------------------------

  const pctCompleted = totalTasks > 0 ? (completedCount / totalTasks) * 100 : 0
  const pctExecuting = totalTasks > 0 ? (executingCount / totalTasks) * 100 : 0
  const pctFailed = totalTasks > 0 ? (failedCount / totalTasks) * 100 : 0
  const pctCancelled = totalTasks > 0 ? (cancelledCount / totalTasks) * 100 : 0

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" onClick={() => navigate(`/plans/${planId}`)}>
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-1">
            <span className="px-2 py-0.5 bg-yellow-500/20 border border-yellow-500/40 text-yellow-300 rounded text-xs font-semibold">
              P{planId}
            </span>
            <h1 className="text-2xl font-bold text-white">Execution Monitor</h1>
          </div>
          <div className="flex items-center gap-2">
            {plan?.name && (
              <span className="text-sm text-[var(--color-text-secondary)]">{plan.name}</span>
            )}
            <span className="px-2 py-0.5 rounded text-xs font-medium bg-blue-500/10 border border-blue-500/30 text-blue-400">
              {getPlanningStrategyName(plan?.planning_strategy || 0)}
            </span>
            <span className="px-2 py-0.5 rounded text-xs font-medium bg-purple-500/10 border border-purple-500/30 text-purple-400">
              {getAllocationStrategyName(plan?.allocation_strategy || 0)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {wallElapsed > 0 && (
            <span className="text-sm font-mono text-[var(--color-text-secondary)]">
              <Clock className="w-3.5 h-3.5 inline mr-1" />
              {formatElapsed(wallElapsed)}
            </span>
          )}
          <StatusBadge status={overallStatus} />
          {isPreview && (
            <Button
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
              className="flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              {startMutation.isPending ? 'Starting...' : 'Start Execution'}
            </Button>
          )}
          {overallStatus === 'failed' && (
            <Button
              onClick={() => copyAndRetryMutation.mutate()}
              disabled={copyAndRetryMutation.isPending}
              className="flex items-center gap-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/30"
            >
              <RotateCcw className="w-4 h-4" />
              {copyAndRetryMutation.isPending ? 'Copying...' : 'Copy & Retry'}
            </Button>
          )}
        </div>
      </div>

      {/* Progress Strip */}
      <Card className="!p-4">
        {isPreview && (
          <div className="mb-3 px-3 py-2 rounded-lg bg-blue-500/10 border border-blue-500/20 text-sm text-blue-300">
            Review the task queue below, then click <strong>Start Execution</strong> when ready.
          </div>
        )}
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-[var(--color-text-secondary)]">Overall Progress</span>
          <span className="text-sm font-mono text-white">
            {completedCount}/{totalTasks} tasks
          </span>
        </div>
        <div className="h-2.5 bg-blue-500/20 rounded-full overflow-hidden flex">
          {pctCompleted > 0 && (
            <div
              className="h-full bg-emerald-500 transition-all duration-500"
              style={{ width: `${pctCompleted}%` }}
            />
          )}
          {pctExecuting > 0 && (
            <div
              className="h-full bg-amber-500 animate-pulse transition-all duration-500"
              style={{ width: `${pctExecuting}%` }}
            />
          )}
          {pctFailed > 0 && (
            <div
              className="h-full bg-red-500 transition-all duration-500"
              style={{ width: `${pctFailed}%` }}
            />
          )}
          {pctCancelled > 0 && (
            <div
              className="h-full bg-slate-500/60 transition-all duration-500"
              style={{ width: `${pctCancelled}%` }}
            />
          )}
        </div>
        <div className="flex flex-wrap gap-2 mt-3">
          <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">
            {completedCount} completed
          </span>
          <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/15 text-amber-400 border border-amber-500/20">
            {executingCount} executing
          </span>
          <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-blue-500/15 text-blue-400 border border-blue-500/20">
            {pendingCount} pending
          </span>
          {failedCount > 0 && (
            <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/15 text-red-400 border border-red-500/20">
              {failedCount} failed
            </span>
          )}
          {cancelledCount > 0 && (
            <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-slate-500/15 text-slate-400 border border-slate-500/20">
              {cancelledCount} cancelled
            </span>
          )}
        </div>
      </Card>

      {/* ================================================================= */}
      {/* Robot Fleet Table                                                  */}
      {/* ================================================================= */}
      <Card>
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('fleet')}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Bot className="w-4 h-4 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Robot Fleet</h3>
            <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded text-xs">
              {robotRows.length} robots
            </span>
          </div>
          {expandedSections.has('fleet')
            ? <ChevronDown className="w-4 h-4 text-[var(--color-text-secondary)]" />
            : <ChevronRight className="w-4 h-4 text-[var(--color-text-secondary)]" />}
        </div>

        {expandedSections.has('fleet') && (
          <div className="mt-4">
            {robotRows.length === 0 ? (
              <div className="text-center py-8 text-[var(--color-text-muted)]">
                <Bot className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No robots assigned to tasks yet</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-[var(--color-text-muted)] uppercase tracking-wider border-b border-border">
                      <th className="pb-3 pr-4">Robot</th>
                      <th className="pb-3 pr-4">Current Task</th>
                      <th className="pb-3 pr-4 text-right">Elapsed</th>
                      <th className="pb-3 pr-4 text-center">Progress</th>
                      <th className="pb-3 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {robotRows.map(row => {
                      const pctOk = row.totalCount > 0 ? (row.succeededCount / row.totalCount) * 100 : 0
                      const pctBad = row.totalCount > 0 ? ((row.failedCount + row.cancelledCount) / row.totalCount) * 100 : 0

                      return (
                      <tr
                        key={row.robotId}
                        className="border-b border-border-subtle hover:bg-surface-overlay/50 cursor-pointer transition-colors"
                        onClick={() => setSelectedRobot(row.robotId)}
                      >
                        {/* Robot */}
                        <td className="py-3 pr-4">
                          <div className="flex items-center gap-2">
                            <Bot className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                            <span className="font-mono text-cyan-300">{row.robotId}</span>
                            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-surface-elevated/50 border border-border text-[var(--color-text-muted)]">
                              {row.robotType}
                            </span>
                          </div>
                        </td>

                        {/* Current Task */}
                        <td className="py-3 pr-4 max-w-xs">
                          {row.currentTask ? (
                            <div className="flex items-center gap-2">
                              <TaskIdChip
                                taskId={row.currentTask.task_id}
                                task={row.currentTask}
                                colorClass="text-amber-400"
                              />
                              <span className="text-[var(--color-text)] truncate">
                                {row.currentTask.description.length > 40
                                  ? row.currentTask.description.slice(0, 40) + '…'
                                  : row.currentTask.description}
                              </span>
                            </div>
                          ) : (
                            <span className="text-[var(--color-text-muted)] italic">
                              {row.finished && row.hasFailed
                                ? `${row.failedCount} failed, ${row.cancelledCount} cancelled`
                                : row.finished
                                  ? 'All tasks complete'
                                  : 'Waiting for deps'}
                            </span>
                          )}
                        </td>

                        {/* Elapsed */}
                        <td className="py-3 pr-4 text-right font-mono text-xs">
                          {row.currentTask && elapsedTimes.has(row.currentTask.task_id) ? (
                            <span className="text-amber-300">
                              {formatElapsed(elapsedTimes.get(row.currentTask.task_id)!)}
                            </span>
                          ) : (
                            <span className="text-[var(--color-text-muted)]">—</span>
                          )}
                        </td>

                        {/* Progress */}
                        <td className="py-3 pr-4">
                          <div className="flex items-center justify-center gap-2">
                            <div className="w-20 h-1.5 bg-surface-overlay rounded-full overflow-hidden flex">
                              {pctOk > 0 && (
                                <div
                                  className="h-full bg-emerald-500 transition-all duration-500"
                                  style={{ width: `${pctOk}%` }}
                                />
                              )}
                              {pctBad > 0 && (
                                <div
                                  className="h-full bg-red-500/70 transition-all duration-500"
                                  style={{ width: `${pctBad}%` }}
                                />
                              )}
                            </div>
                            <span className="text-xs text-[var(--color-text-secondary)] font-mono w-10 text-right">
                              {row.succeededCount}/{row.totalCount}
                            </span>
                          </div>
                        </td>

                        {/* Status */}
                        <td className="py-3 text-center">
                          {row.currentTask ? (
                            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/15 text-amber-300 border border-amber-500/20">
                              Executing
                            </span>
                          ) : row.finished && row.hasFailed ? (
                            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/15 text-red-400 border border-red-500/20">
                              Failed
                            </span>
                          ) : row.finished ? (
                            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/20">
                              Done
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-slate-500/15 text-slate-400 border border-slate-500/20">
                              Waiting
                            </span>
                          )}
                        </td>
                      </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* ================================================================= */}
      {/* Task Queue                                                         */}
      {/* ================================================================= */}
      <Card>
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('queue')}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-amber-500/20 rounded-lg flex items-center justify-center">
              <Clock className="w-4 h-4 text-amber-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Task Queue</h3>
            <span className="px-2 py-1 bg-amber-500/20 text-amber-300 rounded text-xs">
              {pendingCount} pending
            </span>
          </div>
          {expandedSections.has('queue')
            ? <ChevronDown className="w-4 h-4 text-[var(--color-text-secondary)]" />
            : <ChevronRight className="w-4 h-4 text-[var(--color-text-secondary)]" />}
        </div>

        {expandedSections.has('queue') && (
          <div className="mt-4">
            {pendingTasksSorted.length === 0 ? (
              <div className="text-center py-6 text-[var(--color-text-muted)]">
                <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>{totalTasks === 0 ? 'Loading tasks…' : 'All tasks dispatched'}</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-[var(--color-text-muted)] uppercase tracking-wider border-b border-border">
                      <th className="pb-3 pr-4">Task</th>
                      <th className="pb-3 pr-4">Description</th>
                      <th className="pb-3 pr-4">Assigned Robot</th>
                      <th className="pb-3 pr-4">Blocked By</th>
                      <th className="pb-3 text-center">Ready</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pendingTasksSorted.map(task => {
                      const ready = depsReady(task)
                      const blockers = task.dependency_task_ids.filter(
                        d => taskMap.get(d)?.status !== 'completed'
                      )
                      return (
                        <tr key={task.task_id} className="border-b border-border-subtle hover:bg-surface-overlay/50 transition-colors">
                          <td className="py-2.5 pr-4">
                            <TaskIdChip taskId={task.task_id} task={task} />
                          </td>
                          <td className="py-2.5 pr-4 text-[var(--color-text)] max-w-sm truncate">
                            {task.description}
                          </td>
                          <td className="py-2.5 pr-4">
                            {task.robot_id ? (
                              <span className="font-mono text-xs text-cyan-400">{task.robot_id}</span>
                            ) : (
                              <span className="text-xs text-[var(--color-text-muted)]">—</span>
                            )}
                          </td>
                          <td className="py-2.5 pr-4">
                            {blockers.length > 0 ? (
                              <div className="flex flex-wrap gap-1">
                                {blockers.map(d => (
                                  <TaskIdChip key={d} taskId={d} task={taskMap.get(d)} colorClass="text-[var(--color-text-secondary)]" />
                                ))}
                              </div>
                            ) : (
                              <span className="text-xs text-[var(--color-text-muted)]">—</span>
                            )}
                          </td>
                          <td className="py-2.5 text-center">
                            {ready ? (
                              <CheckCircle className="w-4 h-4 text-emerald-400 mx-auto" />
                            ) : (
                              <Circle className="w-4 h-4 text-[var(--color-text-muted)] mx-auto" />
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* ================================================================= */}
      {/* Completed / Failed Tasks                                           */}
      {/* ================================================================= */}
      <Card>
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('completed')}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Completed & Failed Tasks</h3>
            {completedCount > 0 && (
              <span className="px-2 py-1 bg-emerald-500/20 text-emerald-300 rounded text-xs">
                {completedCount} completed
              </span>
            )}
            {(failedCount + cancelledCount) > 0 && (
              <span className="px-2 py-1 bg-red-500/20 text-red-300 rounded text-xs">
                {failedCount + cancelledCount} failed
              </span>
            )}
          </div>
          {expandedSections.has('completed')
            ? <ChevronDown className="w-4 h-4 text-[var(--color-text-secondary)]" />
            : <ChevronRight className="w-4 h-4 text-[var(--color-text-secondary)]" />}
        </div>

        {expandedSections.has('completed') && (
          <div className="mt-4">
            {completedTasksSorted.length === 0 ? (
              <div className="text-center py-6 text-[var(--color-text-muted)]">
                <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No completed tasks yet</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-[var(--color-text-muted)] uppercase tracking-wider border-b border-border">
                      <th className="pb-3 pr-4">Task</th>
                      <th className="pb-3 pr-4">Robot</th>
                      <th className="pb-3 pr-4">Description</th>
                      <th className="pb-3 pr-4">Result</th>
                      <th className="pb-3 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {completedTasksSorted.map(task => (
                      <tr
                        key={task.task_id}
                        className={cn(
                          'border-b border-border-subtle hover:bg-surface-overlay/50 transition-colors',
                          task.status === 'failed' && 'border-l-2 border-l-red-500'
                        )}
                      >
                        <td className="py-2.5 pr-4">
                          <TaskIdChip
                            taskId={task.task_id}
                            task={task}
                            colorClass={task.status === 'failed' ? 'text-red-400' : 'text-emerald-400'}
                          />
                        </td>
                        <td className="py-2.5 pr-4">
                          {task.robot_id ? (
                            <span className="font-mono text-xs text-cyan-400">{task.robot_id}</span>
                          ) : (
                            <span className="text-xs text-[var(--color-text-muted)]">—</span>
                          )}
                        </td>
                        <td className="py-2.5 pr-4 text-[var(--color-text)] max-w-xs truncate">
                          {task.description}
                        </td>
                        <td className="py-2.5 pr-4 max-w-sm">
                          {task.result ? (
                            <details className="group">
                              <summary className="text-xs text-[var(--color-text-secondary)] cursor-pointer hover:text-[var(--color-text)] transition-colors truncate max-w-[200px]">
                                {task.result.length > 60 ? task.result.slice(0, 60) + '…' : task.result}
                              </summary>
                              <pre className="mt-2 text-xs text-[var(--color-text)] whitespace-pre-wrap font-mono bg-surface/80 rounded p-2 border border-border">
                                {task.result}
                              </pre>
                            </details>
                          ) : (
                            <span className="text-xs text-[var(--color-text-muted)]">—</span>
                          )}
                        </td>
                        <td className="py-2.5 text-center">
                          <StatusBadge status={task.status} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* ================================================================= */}
      {/* DAG Visualization                                                  */}
      {/* ================================================================= */}
      <Card>
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('dag')}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-violet-500/20 rounded-lg flex items-center justify-center">
              <Activity className="w-4 h-4 text-violet-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Task Dependencies</h3>
          </div>
          {expandedSections.has('dag')
            ? <ChevronDown className="w-4 h-4 text-[var(--color-text-secondary)]" />
            : <ChevronRight className="w-4 h-4 text-[var(--color-text-secondary)]" />}
        </div>

        {expandedSections.has('dag') && (
          <div className="mt-4 overflow-hidden">
            <DAGVisualization tasks={tasks} height="500px" />
          </div>
        )}
      </Card>

      {/* ================================================================= */}
      {/* Event Log                                                          */}
      {/* ================================================================= */}
      <Card>
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('log')}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-surface-elevated/30 rounded-lg flex items-center justify-center">
              <ScrollText className="w-4 h-4 text-[var(--color-text-secondary)]" />
            </div>
            <h3 className="text-lg font-semibold text-white">Event Log</h3>
            {eventLog.length > 0 && (
              <span className="px-2 py-1 bg-surface-elevated/50 text-[var(--color-text-secondary)] rounded text-xs">
                {eventLog.length} events
              </span>
            )}
          </div>
          {expandedSections.has('log')
            ? <ChevronDown className="w-4 h-4 text-[var(--color-text-secondary)]" />
            : <ChevronRight className="w-4 h-4 text-[var(--color-text-secondary)]" />}
        </div>

        {expandedSections.has('log') && (
          <div className="mt-4 max-h-64 overflow-y-auto">
            {eventLog.length === 0 ? (
              <div className="text-center py-6 text-[var(--color-text-muted)]">
                <ScrollText className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No events yet — waiting for task state changes</p>
              </div>
            ) : (
              <div className="space-y-1 font-mono text-xs">
                {eventLog.map((entry, i) => (
                  <div
                    key={i}
                    className={cn(
                      'flex gap-3 py-1.5 px-2 rounded',
                      entry.level === 'error' && 'bg-red-500/5',
                      entry.level === 'success' && 'bg-emerald-500/5',
                    )}
                  >
                    <span className="text-[var(--color-text-muted)] shrink-0">
                      {formatTimestamp(entry.ts)}
                    </span>
                    <span className={cn(
                      entry.level === 'error' && 'text-red-400',
                      entry.level === 'success' && 'text-emerald-400',
                      entry.level === 'info' && 'text-[var(--color-text)]',
                    )}>
                      {entry.message}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* ================================================================= */}
      {/* Robot Execution Modal                                              */}
      {/* ================================================================= */}
      {selectedRobotRow && (
        <RobotExecutionModal
          isOpen={!!selectedRobot}
          onClose={() => setSelectedRobot(null)}
          robotId={selectedRobotRow.robotId}
          robotType={selectedRobotRow.robotType}
          tasks={selectedRobotRow.tasks}
          allTasks={tasks}
          elapsedTimes={elapsedTimes}
        />
      )}
    </div>
  )
}
