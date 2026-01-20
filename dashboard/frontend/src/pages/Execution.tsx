import { useCallback, useMemo, useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MarkerType,
  Position,
} from 'reactflow'
import 'reactflow/dist/style.css'
import {
  ArrowLeft,
  CheckCircle,
  Clock,
  AlertCircle,
  Circle,
  Bot,
  Play,
  Pause,
  RotateCcw,
  ChevronDown,
  ChevronRight,
  MessageSquare,
  Zap,
  Server
} from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { StatusBadge } from '../components/common/StatusBadge'
import { plansApi, tasksApi, robotsApi, methodsApi } from '../lib/api'
import { cn, getPlanningStrategyName, getAllocationStrategyName, setMethodData } from '../lib/utils'
import type { Task, Robot } from '../types'

function TaskNode({ data }: { data: Task & { color: string } }) {
  const statusIcon = {
    completed: <CheckCircle className="w-4 h-4 text-emerald-400" />,
    in_progress: <Clock className="w-4 h-4 text-amber-400 animate-pulse" />,
    failed: <AlertCircle className="w-4 h-4 text-red-400" />,
    pending: <Circle className="w-4 h-4 text-blue-400" />,
  }

  return (
    <div className={cn(
      'px-4 py-3 rounded-lg border-2 min-w-[200px] max-w-[280px]',
      'bg-slate-900/90 backdrop-blur-sm',
      data.status === 'completed' && 'border-emerald-500/50',
      data.status === 'in_progress' && 'border-amber-500/50 shadow-lg shadow-amber-500/20',
      data.status === 'failed' && 'border-red-500/50',
      data.status === 'pending' && 'border-slate-600',
    )}>
      <div className="flex items-start gap-2">
        {statusIcon[data.status as keyof typeof statusIcon] || statusIcon.pending}
        <div className="flex-1 min-w-0">
          <p className="text-xs font-mono text-slate-400 mb-1">
            Task #{data.task_id}
          </p>
          <p className="text-sm text-white font-medium leading-tight">
            {data.description.length > 50 
              ? data.description.substring(0, 50) + '...' 
              : data.description}
          </p>
          {data.robot_id && (
            <p className="text-xs text-cyber-400 mt-1.5 font-mono">
              @{data.robot_id}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

const nodeTypes = {
  task: TaskNode,
}

export function Execution() {
  const { planId } = useParams<{ planId: string }>()
  const navigate = useNavigate()
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['current', 'queue']))

  const { data: plan } = useQuery({
    queryKey: ['plan', planId],
    queryFn: () => plansApi.get(Number(planId)),
    enabled: !!planId,
  })

  const { data: tasks = [] } = useQuery({
    queryKey: ['tasks', planId],
    queryFn: () => tasksApi.list({ plan_id: Number(planId) }),
    enabled: !!planId,
    refetchInterval: 1000, // Poll for updates every second
  })

  const { data: robots = [] } = useQuery({
    queryKey: ['robots'],
    queryFn: robotsApi.list,
    refetchInterval: 2000,
  })

  // Load method data for strategy name lookups
  const { data: planners = [] } = useQuery({
    queryKey: ['planners'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'planner')),
  })

  const { data: allocators = [] } = useQuery({
    queryKey: ['allocators'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'allocator')),
  })

  // Update method data for name lookups
  useEffect(() => {
    setMethodData(planners, allocators)
  }, [planners, allocators])

  // Group tasks by status
  const taskGroups = useMemo(() => {
    const groups = {
      executing: tasks.filter(t => t.status === 'in_progress'),
      pending: tasks.filter(t => t.status === 'pending'),
      completed: tasks.filter(t => t.status === 'completed'),
      failed: tasks.filter(t => t.status === 'failed'),
    }
    return groups
  }, [tasks])

  // Calculate overall progress
  const totalTasks = tasks.length
  const completedTasks = taskGroups.completed.length
  const progress = totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0

  // Get robot status for executing tasks
  const getRobotForTask = (task: Task) => {
    return robots.find(r => r.robot_id === task.robot_id)
  }

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(section)) {
      newExpanded.delete(section)
    } else {
      newExpanded.add(section)
    }
    setExpandedSections(newExpanded)
  }

  // Build the DAG graph
  const { nodes, edges } = useMemo(() => {
    if (!tasks.length) return { nodes: [], edges: [] }

    // Calculate levels for each task based on dependencies
    const levels = new Map<number, number>()
    const taskMap = new Map(tasks.map(t => [t.task_id, t]))

    const getLevel = (taskId: number, visited = new Set<number>()): number => {
      if (visited.has(taskId)) return 0
      visited.add(taskId)

      if (levels.has(taskId)) return levels.get(taskId)!

      const task = taskMap.get(taskId)
      if (!task || task.dependency_task_ids.length === 0) {
        levels.set(taskId, 0)
        return 0
      }

      const maxDepLevel = Math.max(
        ...task.dependency_task_ids.map(depId => getLevel(depId, visited))
      )
      const level = maxDepLevel + 1
      levels.set(taskId, level)
      return level
    }

    tasks.forEach(t => getLevel(t.task_id))

    // Group tasks by level
    const levelGroups = new Map<number, Task[]>()
    tasks.forEach(task => {
      const level = levels.get(task.task_id) || 0
      if (!levelGroups.has(level)) levelGroups.set(level, [])
      levelGroups.get(level)!.push(task)
    })

    // Create nodes
    const nodes: Node[] = []
    levelGroups.forEach((levelTasks, level) => {
      levelTasks.forEach((task, idx) => {
        nodes.push({
          id: String(task.task_id),
          type: 'task',
          position: {
            x: level * 320,
            y: idx * 140
          },
          data: { ...task },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        })
      })
    })

    // Create edges
    const edges: Edge[] = []
    tasks.forEach(task => {
      task.dependency_task_ids.forEach(depId => {
        edges.push({
          id: `${depId}-${task.task_id}`,
          source: String(depId),
          target: String(task.task_id),
          markerEnd: { type: MarkerType.ArrowClosed },
          style: {
            stroke: '#00d9ff',
            strokeWidth: 2,
          },
          animated: task.status === 'in_progress',
        })
      })
    })

    return { nodes, edges }
  }, [tasks])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" onClick={() => navigate('/plans')}>
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">Plan #{planId} Execution Monitor</h1>
          <p className="text-slate-400">
            {getPlanningStrategyName(plan?.planning_strategy || 0)} planning • {getAllocationStrategyName(plan?.allocation_strategy || 0)} allocation
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={
            progress === 100 ? 'completed' :
            completedTasks > 0 ? 'in_progress' :
            'pending'
          } />
          <span className="text-sm text-slate-400">
            {completedTasks > 0 ?
              `${completedTasks}/${totalTasks} tasks` :
              'Starting execution...'
            }
          </span>
        </div>
      </div>

      {/* Overall Progress */}
      <Card className="!p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm text-slate-400">Overall Progress</span>
          <span className="text-sm font-mono text-white">
            {completedTasks}/{totalTasks} tasks ({Math.round(progress)}%)
          </span>
        </div>
        <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyber-500 to-violet-500 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs text-slate-500">
          <span>{taskGroups.pending.length} pending</span>
          <span>{taskGroups.executing.length} executing</span>
          <span>{taskGroups.completed.length} completed</span>
          <span>{taskGroups.failed.length} failed</span>
        </div>
      </Card>

      {/* Current Execution Status */}
      <Card>
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('current')}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Zap className="w-4 h-4 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Current Execution</h3>
            <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded text-xs">
              {taskGroups.executing.length} active
            </span>
          </div>
          {expandedSections.has('current') ?
            <ChevronDown className="w-4 h-4 text-slate-400" /> :
            <ChevronRight className="w-4 h-4 text-slate-400" />
          }
        </div>

        {expandedSections.has('current') && (
          <div className="mt-4 space-y-3">
            {taskGroups.executing.length === 0 ? (
              <div className="text-center py-6 text-slate-500">
                {completedTasks > 0 ? (
                  <>
                    <Zap className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>All tasks completed or waiting for dependencies</p>
                  </>
                ) : (
                  <>
                    <div className="w-8 h-8 mx-auto mb-2 border-4 border-slate-600 border-t-cyan-400 rounded-full animate-spin"></div>
                    <p>Starting execution...</p>
                    <p className="text-xs mt-1">Tasks will appear here as they begin executing</p>
                  </>
                )}
              </div>
            ) : (
              taskGroups.executing.map(task => {
                const robot = getRobotForTask(task)
                return (
                  <div key={task.task_id} className="flex items-start gap-4 p-4 bg-slate-800/50 rounded-lg border border-blue-500/30">
                    <div className="flex items-center gap-2">
                      <Bot className="w-5 h-5 text-blue-400 animate-pulse" />
                      <div className="text-xs font-mono text-blue-400">#{task.task_id}</div>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <p className="text-white font-medium">{task.description}</p>
                        <div className="flex items-center gap-1 px-2 py-1 bg-blue-500/20 rounded text-xs">
                          <Clock className="w-3 h-3" />
                          Executing
                        </div>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-slate-400">
                        {robot && (
                          <div className="flex items-center gap-2">
                            <Server className="w-4 h-4" />
                            <span>Robot: <span className="text-blue-300 font-mono">@{robot.robot_id}</span></span>
                            <span className="text-slate-600">•</span>
                            <span>Type: {robot.robot_type}</span>
                          </div>
                        )}
                        {task.goal_id && (
                          <>
                            <span className="text-slate-600">•</span>
                            <span>Goal: #{task.goal_id}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        )}
      </Card>

      {/* Task Queue */}
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
              {taskGroups.pending.length} pending
            </span>
          </div>
          {expandedSections.has('queue') ?
            <ChevronDown className="w-4 h-4 text-slate-400" /> :
            <ChevronRight className="w-4 h-4 text-slate-400" />
          }
        </div>

        {expandedSections.has('queue') && (
          <div className="mt-4 space-y-2">
            {taskGroups.pending.length === 0 ? (
              <div className="text-center py-6 text-slate-500">
                {completedTasks === 0 ? (
                  <>
                    <div className="w-8 h-8 mx-auto mb-2 border-4 border-slate-600 border-t-amber-400 rounded-full animate-spin"></div>
                    <p>Loading task queue...</p>
                    <p className="text-xs mt-1">Tasks will appear here once execution begins</p>
                  </>
                ) : (
                  <>
                    <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>All tasks have been assigned</p>
                  </>
                )}
              </div>
            ) : (
              taskGroups.pending.map(task => (
                <div key={task.task_id} className="flex items-center gap-4 p-3 bg-slate-800/30 rounded-lg border border-slate-700/50">
                  <div className="flex items-center gap-2">
                    <Circle className="w-4 h-4 text-slate-400" />
                    <span className="text-xs font-mono text-slate-400">#{task.task_id}</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-slate-300">{task.description}</p>
                    <div className="flex items-center gap-4 text-xs text-slate-500 mt-1">
                      {task.robot_type && <span>Requires: {task.robot_type}</span>}
                      {task.goal_id && <span>Goal: #{task.goal_id}</span>}
                      {task.dependency_task_ids.length > 0 && (
                        <span>Depends on: {task.dependency_task_ids.join(', ')}</span>
                      )}
                    </div>
                  </div>
                  <div className="text-xs text-slate-500">
                    Waiting for dependencies
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </Card>

      {/* Completed Tasks */}
      <Card>
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('completed')}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Completed Tasks</h3>
            <span className="px-2 py-1 bg-emerald-500/20 text-emerald-300 rounded text-xs">
              {taskGroups.completed.length} completed
            </span>
          </div>
          {expandedSections.has('completed') ?
            <ChevronDown className="w-4 h-4 text-slate-400" /> :
            <ChevronRight className="w-4 h-4 text-slate-400" />
          }
        </div>

        {expandedSections.has('completed') && (
          <div className="mt-4 space-y-3">
            {taskGroups.completed.length === 0 ? (
              <div className="text-center py-6 text-slate-500">
                <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No completed tasks yet</p>
              </div>
            ) : (
              taskGroups.completed.map(task => {
                const robot = getRobotForTask(task)
                return (
                  <div key={task.task_id} className="p-4 bg-slate-800/30 rounded-lg border border-emerald-500/20">
                    <div className="flex items-start gap-4">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="w-5 h-5 text-emerald-400" />
                        <span className="text-xs font-mono text-emerald-400">#{task.task_id}</span>
                      </div>
                      <div className="flex-1">
                        <p className="text-white font-medium mb-2">{task.description}</p>
                        <div className="flex items-center gap-4 text-sm text-slate-400 mb-3">
                          {robot && (
                            <div className="flex items-center gap-2">
                              <Server className="w-4 h-4" />
                              <span>Robot: <span className="text-emerald-300 font-mono">@{robot.robot_id}</span></span>
                            </div>
                          )}
                          {task.goal_id && <span>Goal: #{task.goal_id}</span>}
                        </div>

                        {/* Execution Result */}
                        {task.result && (
                          <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50">
                            <div className="flex items-center gap-2 mb-2">
                              <MessageSquare className="w-4 h-4 text-slate-400" />
                              <span className="text-sm font-medium text-slate-300">Execution Result</span>
                            </div>
                            <div className="text-sm text-slate-300 whitespace-pre-wrap font-mono">
                              {task.result}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        )}
      </Card>

      {/* DAG Visualization */}
      <Card>
        <div
          className="flex items-center justify-between cursor-pointer mb-4"
          onClick={() => toggleSection('dag')}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-violet-500/20 rounded-lg flex items-center justify-center">
              <Bot className="w-4 h-4 text-violet-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Task Dependencies</h3>
          </div>
          {expandedSections.has('dag') ?
            <ChevronDown className="w-4 h-4 text-slate-400" /> :
            <ChevronRight className="w-4 h-4 text-slate-400" />
          }
        </div>

        {expandedSections.has('dag') && (
          <div className="h-96">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              fitView
              minZoom={0.3}
              maxZoom={1.5}
              defaultViewport={{ x: 50, y: 50, zoom: 0.8 }}
            >
              <Background color="#1e293b" gap={24} />
              <Controls className="!bg-slate-800 !border-slate-700 !rounded-lg" />
            </ReactFlow>
          </div>
        )}
      </Card>

      {/* Failed Tasks */}
      {taskGroups.failed.length > 0 && (
        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 bg-red-500/20 rounded-lg flex items-center justify-center">
              <AlertCircle className="w-4 h-4 text-red-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Failed Tasks</h3>
            <span className="px-2 py-1 bg-red-500/20 text-red-300 rounded text-xs">
              {taskGroups.failed.length} failed
            </span>
          </div>

          <div className="space-y-3">
            {taskGroups.failed.map(task => {
              const robot = getRobotForTask(task)
              return (
                <div key={task.task_id} className="p-4 bg-slate-800/30 rounded-lg border border-red-500/20">
                  <div className="flex items-start gap-4">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-5 h-5 text-red-400" />
                      <span className="text-xs font-mono text-red-400">#{task.task_id}</span>
                    </div>
                    <div className="flex-1">
                      <p className="text-white font-medium mb-2">{task.description}</p>
                      <div className="flex items-center gap-4 text-sm text-slate-400 mb-3">
                        {robot && (
                          <div className="flex items-center gap-2">
                            <Server className="w-4 h-4" />
                            <span>Robot: <span className="text-red-300 font-mono">@{robot.robot_id}</span></span>
                          </div>
                        )}
                        {task.goal_id && <span>Goal: #{task.goal_id}</span>}
                      </div>

                      {task.result && (
                        <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50">
                          <div className="flex items-center gap-2 mb-2">
                            <MessageSquare className="w-4 h-4 text-slate-400" />
                            <span className="text-sm font-medium text-slate-300">Error Details</span>
                          </div>
                          <div className="text-sm text-red-300 whitespace-pre-wrap font-mono">
                            {task.result}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </Card>
      )}
    </div>
  )
}
