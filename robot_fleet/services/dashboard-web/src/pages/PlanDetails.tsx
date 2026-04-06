import React, { useState, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import {
  ArrowLeft, Wand2, GitBranch, Link, Bot, Target, Play, Edit, Check, CheckCircle, X, Zap, Download, Loader2, Users, Plus, Trash2
} from 'lucide-react'
// Graphviz is imported dynamically below

import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { DAGVisualization } from '../components/common/DAGVisualization'
import { MethodDetailModal } from './Planners'
import type { Robot } from '../types'

// Import AllocatePlanForm from Plans page
// Note: This creates a circular import issue, so we'll inline the allocation logic instead
import { plansApi, robotsApi, goalsApi, methodsApi, tasksApi, useRealtimeUpdates, type MethodSummary } from '../lib/api'
import { cn, capitalize, getPlanningStrategyName, getAllocationStrategyName, getPlanningMethodId, getAllocationMethodId, setMethodData } from '../lib/utils'
import JSZip from 'jszip'

// =============================================================================
// Task List Component
// =============================================================================

function VerticalTaskList({
  tasks,
  onEdit,
  onDelete,
}: {
  tasks: any[]
  onEdit: (task: any) => void
  onDelete: (taskId: number) => void
}) {
  return (
    <div className="space-y-4">
      {tasks.map((task, index) => (
        <Card key={task.task_id || index} className="p-6 hover:bg-surface-overlay/50 transition-colors">
          <div className="space-y-4">
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <div className="flex items-center space-x-3">
                  <span className="text-xl font-bold text-white">Task {task.task_id}</span>
                  <div className="flex items-center space-x-2">
                    <span className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-semibold rounded-lg">
                      {task.robot_type}
                    </span>
                    {task.robot_id && (
                      <span className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-semibold rounded-lg">
                        {task.robot_id}
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-[var(--color-text)] text-base leading-relaxed">{task.description}</div>
              </div>
              <div className="flex flex-col items-end space-y-2">
                {task.status && (
                  <span className="px-3 py-1 bg-surface-elevated border border-border text-[var(--color-text)] text-sm font-medium rounded-lg">
                    {task.status}
                  </span>
                )}
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm" onClick={() => onEdit(task)} disabled={!task.task_id}>
                    <Edit className="w-4 h-4" />
                    Edit
                  </Button>
                  <Button variant="danger" size="sm" onClick={() => onDelete(task.task_id)} disabled={!task.task_id}>
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </Button>
                </div>
              </div>
            </div>

            {task.dependency_task_ids && task.dependency_task_ids.length > 0 && (
              <div className="bg-surface-overlay/40 p-3 rounded-lg border border-border">
                <div className="flex items-center space-x-2 mb-2">
                  <Link className="w-4 h-4 text-[var(--color-text-secondary)]" />
                  <span className="text-sm font-medium text-[var(--color-text)]">Dependencies</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {task.dependency_task_ids.map((depId: number) => (
                    <span key={depId} className="px-3 py-1 bg-orange-500/20 border border-orange-500/40 text-orange-300 text-sm font-semibold rounded-lg">
                      Task {depId}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
      ))}
    </div>
  )
}


// =============================================================================
// Main Component
// =============================================================================

export default function PlanDetails() {
  console.log('PlanDetails: Component rendering')

  const { planId } = useParams<{ planId: string }>()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [selectedMethod, setSelectedMethod] = useState<{ id: number; type: 'planner' | 'allocator' | null } | null>(null)
  const [allocatePlanId, setAllocatePlanId] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'tasks' | 'prompts' | 'artifacts' | 'goals'>('overview')
  const [taskView, setTaskView] = useState<'vertical' | 'dag'>('vertical')
  const [promptsView, setPromptsView] = useState<'planning' | 'allocation'>('planning')
  const [artifactsView, setArtifactsView] = useState<'planning' | 'allocation'>('planning')
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [justUpdated, setJustUpdated] = useState(false)

  const [isTaskEditorOpen, setIsTaskEditorOpen] = useState(false)
  const [taskEditorMode, setTaskEditorMode] = useState<'create' | 'edit'>('create')
  const [editingTaskId, setEditingTaskId] = useState<number | null>(null)
  const [taskFormDescription, setTaskFormDescription] = useState('')
  const [taskFormGoalId, setTaskFormGoalId] = useState<number | ''>('')
  const [taskFormRobotId, setTaskFormRobotId] = useState<string>('') // "" = unassigned
  const [taskFormDependencyIds, setTaskFormDependencyIds] = useState<number[]>([])
  const [deleteTaskId, setDeleteTaskId] = useState<number | null>(null)

  // Enable real-time updates
  const { isConnected } = useRealtimeUpdates()


  // Handle URL query parameters for method modal
  useEffect(() => {
    const method = searchParams.get('method')
    const methodType = searchParams.get('method_type') as 'planner' | 'allocator' | null
    if (method) {
      const methodId = parseInt(method, 10)
      if (!isNaN(methodId)) {
        setSelectedMethod({ id: methodId, type: methodType })
      }
    } else {
      setSelectedMethod(null)
    }
  }, [searchParams])
  const queryClient = useQueryClient()

  console.log('PlanDetails: planId =', planId)

  const { data: plan, isLoading, error, isFetching } = useQuery({
    queryKey: ['plan', planId],
    queryFn: () => plansApi.get(parseInt(planId!)),
    enabled: !!planId,
  })

  // Fetch allocation status separately
  const { data: allocationStatus } = useQuery({
    queryKey: ['plan-status', planId],
    queryFn: () => plansApi.getStatus(parseInt(planId!)),
    enabled: !!planId,
    // No refetchInterval - using WebSocket real-time updates
  })

  // Fetch robot statuses for execution readiness check
  const { data: robots } = useQuery({
    queryKey: ['robots'],
    queryFn: () => robotsApi.list(),
    // No refetchInterval - using WebSocket real-time updates
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

  // Fetch robot health statuses for accurate connectivity checks
  const { data: robotHealth } = useQuery({
    queryKey: ['robot-health'],
    queryFn: async () => {
      try {
        const response = await fetch('/api/robots/health/all')
        const data = await response.json()
        const healthMap: Record<string, { reachable: boolean }> = {}
        for (const health of data.robots || []) {
          healthMap[health.robot_id] = { reachable: health.reachable }
        }
        return healthMap
      } catch (error) {
        console.error('Failed to fetch robot health:', error)
        return {}
      }
    },
    // No refetchInterval - using WebSocket real-time updates
  })

  const { data: goals = [] } = useQuery({
    queryKey: ['goals'],
    queryFn: goalsApi.list,
  })

  const openCreateTask = () => {
    const defaultGoal = (plan as any)?.goal_ids?.[0]
    setTaskEditorMode('create')
    setEditingTaskId(null)
    setTaskFormDescription('')
    setTaskFormGoalId(typeof defaultGoal === 'number' ? defaultGoal : '')
    setTaskFormRobotId('')
    setTaskFormDependencyIds([])
    setIsTaskEditorOpen(true)
  }

  const openEditTask = (task: any) => {
    setTaskEditorMode('edit')
    setEditingTaskId(task.task_id)
    setTaskFormDescription(task.description || '')
    setTaskFormGoalId(typeof task.goal_id === 'number' ? task.goal_id : '')
    setTaskFormRobotId(task.robot_id || '')
    setTaskFormDependencyIds(Array.isArray(task.dependency_task_ids) ? task.dependency_task_ids : [])
    setIsTaskEditorOpen(true)
  }

  const requestDeleteTask = (taskId: number) => setDeleteTaskId(taskId)

  const submitTaskEditor = (e: React.FormEvent) => {
    e.preventDefault()
    if (!planId) return
    const parsedPlanId = parseInt(planId, 10)
    if (!parsedPlanId) return
    if (!taskFormDescription.trim()) return
    if (taskFormGoalId === '') return

    const robotIdValue = taskFormRobotId // "" clears/unassigns; non-empty sets
    if (taskEditorMode === 'create') {
      createTaskMutation.mutate({
        description: taskFormDescription.trim(),
        goal_id: taskFormGoalId,
        plan_id: parsedPlanId,
        robot_id: robotIdValue ? robotIdValue : null,
        dependency_task_ids: taskFormDependencyIds,
      })
    } else if (editingTaskId != null) {
      updateTaskMutation.mutate({
        taskId: editingTaskId,
        data: {
          description: taskFormDescription.trim(),
          goal_id: taskFormGoalId,
          robot_id: robotIdValue,
          dependency_task_ids: taskFormDependencyIds,
        },
      })
    }
  }

  // Check if plan is executable (has allocated tasks and robots are available)
  const isPlanExecutable = plan && allocationStatus && robotHealth && robots ? (() => {
    if (allocationStatus.status !== 'fully_allocated') return false

    // Get all robots assigned to tasks in this plan
    const planTasks = plan.tasks || []
    const assignedTasks = planTasks.filter(t => t.robot_id)
    const assignedRobotIds = [
      ...new Set(
        assignedTasks
          .map((t: any) => t.robot_id)
          .filter((id: unknown): id is string => typeof id === 'string' && id.length > 0)
      ),
    ]

    // Check if all assigned robots are reachable
    return assignedRobotIds.every(robotId => {
      const robot = robots.find(r => r.robot_id === robotId)
      const health = robotHealth[robotId]
      return robot && health?.reachable === true
    })
  })() : false

  const updateMutation = useMutation({
    mutationFn: ({ planId, name, description }: { planId: number; name: string; description: string }) =>
      plansApi.update(planId, { name, description }),
    onSuccess: (updatedPlan) => {
      console.log('Update successful, received:', updatedPlan)
      // Update the query cache with the new plan info
      queryClient.setQueryData(['plans', planId], updatedPlan)
      queryClient.setQueryData(['plan', planId], updatedPlan)
      console.log('Cache updated, invalidating queries...')
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      // Force a refetch to ensure the data is fresh
      queryClient.invalidateQueries({ queryKey: ['plan', planId] })

      // Mark that we just updated
      setJustUpdated(true)
    },
    onError: (error) => {
      console.error('Failed to update plan:', error)
      setIsEditing(false)
      setJustUpdated(false)
    },
  })

  // Allocation mutation
  const allocateMutation = useMutation({
    mutationFn: ({ planId, allocationStrategy }: { planId: number; allocationStrategy: string }) =>
      plansApi.allocate(planId, allocationStrategy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      queryClient.invalidateQueries({ queryKey: ['plan', planId] })
      setAllocatePlanId(null)
    },
  })

  const createTaskMutation = useMutation({
    mutationFn: (payload: { description: string; goal_id: number; plan_id: number; robot_id?: string | null; dependency_task_ids: number[] }) =>
      tasksApi.create(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      queryClient.invalidateQueries({ queryKey: ['plan', planId] })
      queryClient.invalidateQueries({ queryKey: ['plan-status', planId] })
      setIsTaskEditorOpen(false)
      setEditingTaskId(null)
    },
  })

  const updateTaskMutation = useMutation({
    mutationFn: (payload: { taskId: number; data: { description?: string; goal_id?: number; robot_id?: string; dependency_task_ids: number[] } }) =>
      tasksApi.update(payload.taskId, {
        ...payload.data,
        update_dependency_task_ids: true,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      queryClient.invalidateQueries({ queryKey: ['plan', planId] })
      queryClient.invalidateQueries({ queryKey: ['plan-status', planId] })
      setIsTaskEditorOpen(false)
      setEditingTaskId(null)
    },
  })

  const deleteTaskMutation = useMutation({
    mutationFn: (taskIdToDelete: number) => tasksApi.delete(taskIdToDelete),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      queryClient.invalidateQueries({ queryKey: ['plan', planId] })
      queryClient.invalidateQueries({ queryKey: ['plan-status', planId] })
      setDeleteTaskId(null)
    },
  })

  // Exit edit mode only after the query has finished refetching with the new data
  useEffect(() => {
    if (justUpdated && !isFetching) {
      console.log('Query finished refetching, exiting edit mode')
      setIsEditing(false)
      setJustUpdated(false)
    }
  }, [justUpdated, isFetching])

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Bot },
    { id: 'tasks', label: 'Tasks', icon: GitBranch },
    { id: 'goals', label: 'Goals', icon: Target },
    { id: 'prompts', label: 'Prompts', icon: Wand2 },
    { id: 'artifacts', label: 'Artifacts', icon: Link },
  ]

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[var(--color-bg)] p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center py-12">
            <div className="text-[var(--color-text-secondary)]">Loading plan details...</div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[var(--color-bg)] p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center py-12">
            <div className="text-red-400">Error loading plan: {error.message}</div>
          </div>
        </div>
      </div>
    )
  }

  if (!plan) {
    return (
      <div className="min-h-screen bg-[var(--color-bg)] p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center py-12">
            <div className="text-[var(--color-text-secondary)]">Plan not found</div>
          </div>
        </div>
      </div>
    )
  }

  // Download handlers (defined here for access to plan data and svgContent)
  const downloadFile = (content: string, filename: string, mimeType: string = 'text/plain') => {
    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const downloadJSON = (data: any, filename: string) => {
    const jsonString = JSON.stringify(data, null, 2)
    downloadFile(jsonString, filename, 'application/json')
  }

  const downloadSVG = (svgContent: string, filename: string) => {
    downloadFile(svgContent, filename, 'image/svg+xml')
  }

  const downloadDAG = async () => {
    console.log('Download DAG called')

    try {
      // Dynamically import Graphviz and generate SVG on-demand
      const { Graphviz } = await import('@hpcc-js/wasm-graphviz')
      const graphviz = await Graphviz.load()

      // Generate the DOT source (same logic as in DAGVisualization)
      const tasks = plan.tasks || []
      const generateDot = (tasks: any[]) => {
        let dot = `digraph G {
  rankdir=LR;
  bgcolor="#ffffff";
  node [shape=plaintext, fontname="Arial"];
  edge [color="#334155",penwidth=5.0, arrowhead=vee, arrowsize=4.0, headclip=true, tailclip=true, fontname="Arial"];
  graph [splines=spline, nodesep=1.2, ranksep=1.4];
`

        // Add nodes with status-based styling
        tasks.forEach(task => {
          // Status-based colors for light background
          let statusColors = {
            border: '#1e293b',
            bg: '#ffffff',
            text: '#0f172a',
            headerBg: '#f1f5f9'
          }

          switch ((task.status || 'pending').toLowerCase()) {
            case 'completed':
              statusColors = { border: '#047857', bg: '#d1fae5', text: '#064e3b', headerBg: '#a7f3d0' }
              break
            case 'running':
            case 'executing':
              statusColors = { border: '#d97706', bg: '#fef3c7', text: '#92400e', headerBg: '#fde68a' }
              break
            case 'failed':
            case 'error':
              statusColors = { border: '#dc2626', bg: '#fee2e2', text: '#991b1b', headerBg: '#fecaca' }
              break
            case 'pending':
            default:
              statusColors = { border: '#64748b', bg: '#f8fafc', text: '#374151', headerBg: '#f3f4f6' }
              break
          }

          const htmlLabel = `<TABLE BORDER="3" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" BGCOLOR="${statusColors.bg}" COLOR="${statusColors.border}">
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="${statusColors.headerBg}" CELLPADDING="12">
      <FONT COLOR="#1d4ed8" FACE="Arial" SIZE="28"><B>Task ${task.task_id}</B></FONT>
    </TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" CELLPADDING="16">
      <FONT COLOR="${statusColors.text}" FACE="Arial" SIZE="20">${task.description}</FONT>
    </TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" CELLPADDING="12">
      <TABLE BORDER="0" CELLSPACING="10">
        <TR>
          <TD BGCOLOR="#7c3aed" CELLPADDING="10">
            <FONT COLOR="#ffffff" FACE="Arial" SIZE="16"><B>${task.robot_type || 'unknown'}</B></FONT>
          </TD>
          <TD BGCOLOR="#0d9488" CELLPADDING="10">
            <FONT COLOR="#ffffff" FACE="Arial" SIZE="16"><B>${task.robot_id || 'unassigned'}</B></FONT>
          </TD>
        </TR>
      </TABLE>
    </TD>
  </TR>
</TABLE>`

          dot += `  ${task.task_id} [label=<${htmlLabel}>];\n`
        })

        // Add edges with status-based colors
        tasks.forEach(task => {
          if (task.dependency_task_ids && task.dependency_task_ids.length > 0) {
            task.dependency_task_ids.forEach((depId: number) => {
              // Use a dark, visible edge color for better contrast on white background
              dot += `  ${depId} -> ${task.task_id} [color="#374151", penwidth="3"];\n`
            })
          }
        })

        dot += '}'
        return dot
      }

      const dotSource = generateDot(tasks)
      const svg = graphviz.dot(dotSource, 'svg')

      if (svg && svg.length > 100 && svg.includes('<svg')) {
        console.log('Generated fresh SVG for download, length:', svg.length)
        downloadSVG(svg, `plan-${planId}-dag.svg`)
      } else {
        console.error('Generated SVG is invalid')
        alert('Failed to generate DAG visualization for download.')
      }

    } catch (error) {
      console.error('Error generating SVG for download:', error)
      alert('Failed to generate DAG visualization. Please try again.')
    }
  }

  const downloadPlanningPrompts = async () => {
    if (plan.planning_prompts && typeof plan.planning_prompts === 'object') {
      const zip = new JSZip()

      if (plan.planning_prompts.system) {
        zip.file('system_prompt.txt', plan.planning_prompts.system)
      }
      if (plan.planning_prompts.user) {
        zip.file('user_prompt.txt', plan.planning_prompts.user)
      }

      const content = await zip.generateAsync({ type: 'blob' })
      const url = URL.createObjectURL(content)
      const a = document.createElement('a')
      a.href = url
      a.download = `plan-${planId}-planning-prompts.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  const downloadAllocationPrompts = async () => {
    if (plan.allocation_prompts && typeof plan.allocation_prompts === 'object') {
      const zip = new JSZip()

      if (plan.allocation_prompts.system) {
        zip.file('system_prompt.txt', plan.allocation_prompts.system)
      }
      if (plan.allocation_prompts.user) {
        zip.file('user_prompt.txt', plan.allocation_prompts.user)
      }

      const content = await zip.generateAsync({ type: 'blob' })
      const url = URL.createObjectURL(content)
      const a = document.createElement('a')
      a.href = url
      a.download = `plan-${planId}-allocation-prompts.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  const downloadPlanningArtifacts = () => {
    if (plan.planning_artifacts) {
      downloadJSON(plan.planning_artifacts, `plan-${planId}-planning-artifacts.json`)
    }
  }

  const downloadAllocationArtifacts = () => {
    if (plan.allocation_artifacts) {
      downloadJSON(plan.allocation_artifacts, `plan-${planId}-allocation-artifacts.json`)
    }
  }

  const downloadFullPlan = async () => {
    try {
      const zip = new JSZip()

      // Create directory structure
      const promptsFolder = zip.folder('prompts')
      const planningPromptsFolder = promptsFolder?.folder('planning')
      const allocationPromptsFolder = promptsFolder?.folder('allocation')

      const artifactsFolder = zip.folder('artifacts')

      // Add planning prompts
      if (plan.planning_prompts && typeof plan.planning_prompts === 'object') {
        if (plan.planning_prompts.system) {
          planningPromptsFolder?.file('system_prompt.txt', plan.planning_prompts.system)
        }
        if (plan.planning_prompts.user) {
          planningPromptsFolder?.file('user_prompt.txt', plan.planning_prompts.user)
        }
      }

      // Add allocation prompts
      if (plan.allocation_prompts && typeof plan.allocation_prompts === 'object') {
        if (plan.allocation_prompts.system) {
          allocationPromptsFolder?.file('system_prompt.txt', plan.allocation_prompts.system)
        }
        if (plan.allocation_prompts.user) {
          allocationPromptsFolder?.file('user_prompt.txt', plan.allocation_prompts.user)
        }
      }

      // Add artifacts
      if (plan.planning_artifacts) {
        artifactsFolder?.file('planning_artifacts.json', JSON.stringify(plan.planning_artifacts, null, 2))
      }
      if (plan.allocation_artifacts) {
        artifactsFolder?.file('allocation_artifacts.json', JSON.stringify(plan.allocation_artifacts, null, 2))
      }

      // Generate and add DAG SVG
      try {
        const { Graphviz } = await import('@hpcc-js/wasm-graphviz')
        const graphviz = await Graphviz.load()

        const tasks = plan.tasks || []
        const generateDot = (tasks: any[]) => {
          let dot = `digraph G {
  rankdir=LR;
  bgcolor="#ffffff";
  node [shape=plaintext, fontname="Arial"];
  edge [color="#334155",penwidth=5.0, arrowhead=vee, arrowsize=4.0, headclip=true, tailclip=true, fontname="Arial"];
  graph [splines=spline, nodesep=1.2, ranksep=1.4];
`

          tasks.forEach(task => {
            let statusColors = {
              border: '#1e293b',
              bg: '#ffffff',
              text: '#0f172a',
              headerBg: '#f1f5f9'
            }

            switch ((task.status || 'pending').toLowerCase()) {
              case 'completed':
                statusColors = { border: '#047857', bg: '#d1fae5', text: '#064e3b', headerBg: '#a7f3d0' }
                break
              case 'running':
              case 'executing':
                statusColors = { border: '#d97706', bg: '#fef3c7', text: '#92400e', headerBg: '#fde68a' }
                break
              case 'failed':
              case 'error':
                statusColors = { border: '#dc2626', bg: '#fee2e2', text: '#991b1b', headerBg: '#fecaca' }
                break
              case 'pending':
              default:
                statusColors = { border: '#64748b', bg: '#f8fafc', text: '#374151', headerBg: '#f3f4f6' }
                break
            }

            const htmlLabel = `<TABLE BORDER="3" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" BGCOLOR="${statusColors.bg}" COLOR="${statusColors.border}">
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="${statusColors.headerBg}" CELLPADDING="12">
      <FONT COLOR="#1d4ed8" FACE="Arial" SIZE="28"><B>Task ${task.task_id}</B></FONT>
    </TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" CELLPADDING="16">
      <FONT COLOR="${statusColors.text}" FACE="Arial" SIZE="20">${task.description}</FONT>
    </TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" CELLPADDING="12">
      <TABLE BORDER="0" CELLSPACING="10">
        <TR>
          <TD BGCOLOR="#7c3aed" CELLPADDING="10">
            <FONT COLOR="#ffffff" FACE="Arial" SIZE="16"><B>${task.robot_type || 'unknown'}</B></FONT>
          </TD>
          <TD BGCOLOR="#0d9488" CELLPADDING="10">
            <FONT COLOR="#ffffff" FACE="Arial" SIZE="16"><B>${task.robot_id || 'unassigned'}</B></FONT>
          </TD>
        </TR>
      </TABLE>
    </TD>
  </TR>
</TABLE>`

            dot += `  ${task.task_id} [label=<${htmlLabel}>];\n`
          })

          tasks.forEach(task => {
            if (task.dependency_task_ids && task.dependency_task_ids.length > 0) {
              task.dependency_task_ids.forEach((depId: number) => {
                dot += `  ${depId} -> ${task.task_id} [color="#374151", penwidth="3"];\n`
              })
            }
          })

          dot += '}'
          return dot
        }

        const dotSource = generateDot(tasks)
        const dagSvg = graphviz.dot(dotSource, 'svg')

        if (dagSvg && dagSvg.length > 100 && dagSvg.includes('<svg')) {
          zip.file('dag.svg', dagSvg)
        }
      } catch (svgError) {
        console.warn('Could not generate DAG SVG for zip:', svgError)
      }

      // Generate and download the zip
      const content = await zip.generateAsync({ type: 'blob' })
      const url = URL.createObjectURL(content)
      const a = document.createElement('a')
      a.href = url
      a.download = `plan-${planId}-complete.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

    } catch (error) {
      console.error('Error creating full plan download:', error)
      alert('Failed to create full plan download. Please try individual downloads.')
    }
  }

  return (
    <>
      <div className="min-h-screen bg-[var(--color-bg)] p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                const fromParam = searchParams.get('from')
                if (fromParam?.startsWith('robot-')) {
                  const robotId = fromParam.replace('robot-', '')
                  navigate(`/robots?robot=${robotId}&tab=allocations`)
                } else {
                  navigate('/plans')
                }
              }}
              className="flex items-center space-x-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>{searchParams.get('from')?.startsWith('robot-') ? 'Back to Robot' : 'Back to Plans'}</span>
            </Button>
            <h1 className="text-2xl font-bold text-white">Plan Details</h1>
          </div>

          {/* Execute / Allocate Button */}
          {plan && (
            <>
              {!isPlanExecutable && allocationStatus?.status !== 'fully_allocated' ? (
                <Button
                  onClick={() => setAllocatePlanId(plan.plan_id)}
                  disabled={allocateMutation.isPending}
                  className="flex items-center space-x-2"
                >
                  <Users className="w-4 h-4" />
                  {allocateMutation.isPending ? 'Allocating...' : 'Allocate Robots'}
                </Button>
              ) : (
                <Button
                  onClick={() => navigate(`/plans/${planId}/execute`)}
                  disabled={plan.execution_status === 'completed'}
                  className="flex items-center space-x-2"
                >
                  <Play className="w-4 h-4" />
                  {plan.execution_status === 'executing' ? 'View Execution' :
                   plan.execution_status === 'completed' ? 'Completed' :
                   'View Execution'}
                </Button>
              )}
            </>
          )}
        </div>

        {/* Plan Info Card */}
        <Card className="p-6">
          {/* Top Row: P{ID} + Status + Edit Button */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              {/* P{Plan_id} Badge */}
              <span className="px-3 py-1 bg-yellow-500/20 border border-yellow-500/40 text-yellow-300 rounded-lg text-sm font-semibold">
                P{plan.plan_id}
              </span>

              {/* Planning Method */}
              {plan.planning_strategy && (
                <button
                  className="px-3 py-1.5 bg-blue-500/10 border border-blue-500/20 text-blue-400 rounded-lg text-sm font-semibold hover:bg-blue-500/30 transition-colors"
                  onClick={() => {
                    // Manual planning is not a method-details modal
                    if (plan.planning_strategy === 4) return
                    const methodId = getPlanningMethodId(plan.planning_strategy)
                    navigate(`/plans/${planId}?method_type=planner&method=${methodId}`)
                  }}
                >
                  {getPlanningStrategyName(plan.planning_strategy)}
                </button>
              )}

              {/* Allocation Method */}
              {allocationStatus?.status && allocationStatus.status !== 'unallocated' && plan.allocation_strategy && plan.allocation_strategy !== 4 && (
                <button
                  className="px-3 py-1.5 bg-violet-500/10 border border-violet-500/20 text-violet-400 rounded-lg text-sm font-semibold hover:bg-purple-500/30 transition-colors"
                  onClick={() => {
                    // Manual allocation is not a method-details modal
                    if (plan.allocation_strategy === 5) return
                    const methodId = getAllocationMethodId(plan.allocation_strategy)
                    navigate(`/plans/${planId}?method_type=allocator&method=${methodId}`)
                  }}
                >
                  {getAllocationStrategyName(plan.allocation_strategy)}
                </button>
              )}

              {/* Execution Status */}
              {(() => {
                const tasks = plan.tasks || [];
                const assignedTasks = tasks.filter(t => t.robot_id);
                const assignedRobotIds = [...new Set(assignedTasks.map(t => t.robot_id).filter(Boolean))];
                const allocationStatusValue = allocationStatus?.status || 'unknown';
                const robotsReady = assignedRobotIds.length > 0 && assignedRobotIds.every((robotId) => {
                  const robot = robots?.find(r => r.robot_id === robotId);
                  // @ts-ignore - TypeScript false positive, we check robotHealth exists
                  const health = robotHealth ? robotHealth[robotId] : undefined;
                  return robot && health?.reachable === true;
                });
                const executionReady = allocationStatusValue === 'fully_allocated' && robotsReady;

                return (
                  <span className={`px-3 py-1.5 border rounded-lg text-sm font-semibold ${
                    executionReady ? 'bg-blue-500/10 border-blue-500/20 text-blue-400' :
                    allocationStatusValue === 'fully_allocated' ? 'bg-amber-500/10 border-amber-500/20 text-amber-400' :
                    'bg-red-500/10 border-red-500/20 text-red-400'
                  }`}>
                    {executionReady ? (
                      <>
                        <Zap className="w-4 h-4 mr-1 inline" />
                        Ready to Execute
                      </>
                    ) : allocationStatusValue === 'fully_allocated' ? 'Robots Unavailable' : 'Not Ready'}
                  </span>
                );
              })()}
            </div>

            {/* Edit Button */}
            {isEditing ? (
              <div className="flex items-center space-x-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    setIsEditing(false)
                    setEditName('')
                    setEditDescription('')
                  }}
                  className="text-sm"
                >
                  <X className="w-4 h-4 mr-1" />
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => {
                    updateMutation.mutate({
                      planId: plan.plan_id,
                      name: editName.trim(),
                      description: editDescription.trim()
                    })
                  }}
                  disabled={!editName.trim() || updateMutation.isPending}
                  className="text-sm bg-emerald-600 hover:bg-emerald-700 border border-emerald-500"
                >
                  {updateMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Check className="w-4 h-4 mr-1" />
                      Save
                    </>
                  )}
                </Button>
              </div>
            ) : (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  setIsEditing(true)
                  setEditName(plan.name || '')
                  setEditDescription(plan.description || '')
                }}
                className="text-sm"
              >
                <Edit className="w-4 h-4 mr-1" />
                Edit
              </Button>
            )}
          </div>

          {/* Plan Name and Description */}
          {isEditing ? (
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
                  Plan Name
                </label>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="w-full px-3 py-2 bg-surface-overlay border border-border rounded-lg text-white placeholder-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  placeholder="Enter plan name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
                  Description
                </label>
                <textarea
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                  rows={2}
                  className="w-full px-3 py-2 bg-surface-overlay border border-border rounded-lg text-white placeholder-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-emerald-500 resize-none"
                  placeholder="Enter plan description"
                />
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <h2 className="text-xl font-bold text-white">
                {plan.name}
              </h2>
              <div className="text-[var(--color-text)] text-base leading-relaxed">
                {plan.description}
              </div>
            </div>
          )}

          {/* Bottom Row: Created date + Download Button */}
          <div className="flex items-center justify-between mt-4">
            {plan.created_at && (
              <div className="text-[var(--color-text-muted)] text-xs">
                Created on: {new Date(plan.created_at).toLocaleString()}
              </div>
            )}
            <Button
              variant="primary"
              size="sm"
              onClick={() => {
                console.log('Download Full Plan button clicked')
                if (typeof downloadFullPlan === 'function') {
                  downloadFullPlan()
                } else {
                  console.error('downloadFullPlan function not found')
                }
              }}
              className="text-sm bg-emerald-600 hover:bg-emerald-700 border border-emerald-500 shadow-md font-semibold"
            >
                <Download className="w-4 h-4 mr-2" />
                Download Full Plan (ZIP)
            </Button>
          </div>
        </Card>

        {/* Tabs */}
        <div className="space-y-4">
          <div className="border-b border-border">
            <div className="flex space-x-1">
              {tabs.map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={cn(
                      'px-4 py-2 text-sm font-medium rounded-t-md transition-colors flex items-center space-x-2',
                      activeTab === tab.id
                        ? 'bg-surface-overlay text-white border-b-2 border-blue-500'
                        : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Tab Content */}
          <div className="min-h-[400px]">
            {activeTab === 'overview' && (() => {
              // Calculate execution readiness data outside IIFE for use in both overview and execution readiness

              return (
                <div className="space-y-6">
                  {(() => {
                  // Calculate useful metrics for research scientists
                  const tasks = plan.tasks || [];
                  const assignedTasks = tasks.filter(t => t.robot_id);
                  const unassignedTasks = tasks.filter(t => !t.robot_id);

                  // Check execution readiness - all assigned robots must be online and ready
                  const assignedRobotIds = [...new Set(assignedTasks.map(t => t.robot_id).filter(Boolean))];


                  const robotsReady = assignedRobotIds.length > 0 && assignedRobotIds.every(robotId => {
                    // Check both that robot exists AND is reachable via health check
                    const robot = robots?.find(r => r.robot_id === robotId);
                    const health = robotHealth && robotId ? (robotHealth as any)[robotId] : undefined;
                    return robot && health?.reachable === true;
                  });

                  // Count actually offline robots for better messaging
                  const offlineRobots = assignedRobotIds.filter(robotId => {
                    const robot = robots?.find(r => r.robot_id === robotId);
                    const health = robotHealth && robotId ? (robotHealth as any)[robotId] : undefined;
                    return !robot || health?.reachable !== true;
                  });

                  const allocationStatusValue = allocationStatus?.status || 'unknown';
                  const executionReady = allocationStatusValue === 'fully_allocated' && robotsReady;


                  // Robot utilization stats
                  const robotTypes = [...new Set(tasks.map(t => t.robot_type).filter(Boolean))];
                  const robotIds = [...new Set(assignedTasks.map(t => t.robot_id).filter(Boolean))];

                  // Task distribution by robot type
                  const tasksByType = tasks.reduce((acc, task) => {
                    const type = task.robot_type || 'unassigned';
                    acc[type] = (acc[type] || 0) + 1;
                    return acc;
                  }, {} as Record<string, number>);

                  // Tasks by specific robot ID
                  const tasksByRobotId = assignedTasks.reduce((acc, task) => {
                    const robotId = task.robot_id;
                    if (robotId) {
                      acc[robotId] = (acc[robotId] || 0) + 1;
                    }
                    return acc;
                  }, {} as Record<string, number>);

                  return (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Robot Fleet Utilization */}
                      <Card className="p-6 border-border">
                        <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                            <Bot className="w-4 h-4 text-emerald-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Robot Fleet Utilization</h3>
                        </div>

                        {/* Top metrics in horizontal layout */}
                        <div className="grid grid-cols-3 gap-4 mb-6">
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-emerald-300 mb-1">
                              {robotTypes.length}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Robot Types Used
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-emerald-300 mb-1">
                              {robotIds.length}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Robots Assigned
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-blue-300 mb-1">
                              {tasks.length > 0 ? Math.round((assignedTasks.length / tasks.length) * 100) : 0}%
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Assignment Rate
                            </div>
                          </div>
                        </div>

                        {/* Bottom details with styled sections */}
                        {(robotTypes.length > 0 || robotIds.length > 0) && (
                          <div className="space-y-4">
                            {robotTypes.length > 0 && (
                              <div className="bg-surface-overlay/40 rounded-lg p-3 border border-border/30">
                                <div className="flex items-center space-x-2 mb-3">
                                  <div className="w-5 h-5 bg-surface-elevated/30 rounded border border-border flex items-center justify-center">
                                    <span className="text-xs font-bold text-[var(--color-text)]">T</span>
                                  </div>
                                  <h4 className="text-sm font-semibold text-[var(--color-text)]">Robot Types Used</h4>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  {robotTypes.map(type => (
                                    <span key={type} className="px-3 py-2 bg-surface-elevated border border-border text-[var(--color-text)] text-sm font-medium rounded-lg">
                                      {type}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                            {robotIds.length > 0 && (
                              <div className="bg-surface-overlay/40 rounded-lg p-3 border border-border/30">
                                <div className="flex items-center space-x-2 mb-3">
                                  <div className="w-5 h-5 bg-emerald-500/20 rounded border border-emerald-500/40 flex items-center justify-center">
                                    <span className="text-xs font-bold text-emerald-300">R</span>
                                  </div>
                                  <h4 className="text-sm font-semibold text-emerald-300">Specific Robots Used</h4>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  {robotIds.map(robotId => (
                                    <span key={robotId} className="px-3 py-2 bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-sm font-medium rounded-lg">
                                      {robotId}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </Card>

                      {/* Task Distribution */}
                      <Card className="p-6 border-border">
                        <div>
                          <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                            <GitBranch className="w-4 h-4 text-blue-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Task Distribution</h3>
                        </div>

                        {/* Top metrics in horizontal layout */}
                        <div className="grid grid-cols-3 gap-4 mb-6">
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-blue-300 mb-1">
                              {tasks.length}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Total Tasks
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-emerald-300 mb-1">
                              {assignedTasks.length}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Assigned Tasks
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-amber-300 mb-1">
                              {unassignedTasks.length}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Unassigned Tasks
                            </div>
                          </div>
                        </div>
                          {(Object.keys(tasksByType).length > 1 || Object.keys(tasksByRobotId).length > 0) && (
                            <div className="pt-3 border-t border-border space-y-4">
                              {Object.keys(tasksByType).length > 1 && (
                                <div className="bg-surface-overlay/40 rounded-lg p-3 border border-border/30">
                                  <div className="flex items-center space-x-2 mb-3">
                                    <div className="w-5 h-5 bg-blue-500/20 rounded border border-blue-500/40 flex items-center justify-center">
                                      <span className="text-xs font-bold text-blue-300">T</span>
                                    </div>
                                    <h4 className="text-sm font-semibold text-blue-300">By Robot Type</h4>
                                  </div>
                                  <div className="grid grid-cols-1 gap-2">
                                    {Object.entries(tasksByType)
                                      .sort(([,a], [,b]) => b - a)
                                      .map(([type, count]) => {
                                        const percentage = tasks.length > 0 ? (count / tasks.length) * 100 : 0;
                                        return (
                                          <div key={type} className="flex items-center justify-between p-2 bg-surface-overlay/30 rounded border border-border/20">
                                            <div className="flex items-center space-x-2">
                                              <span className="text-white text-sm font-medium px-2 py-1 bg-surface-elevated/50 rounded">
                                                {type}
                                              </span>
                                              <span className="text-[var(--color-text-secondary)] text-xs">
                                                {percentage.toFixed(0)}%
                                              </span>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                              <div className="w-12 bg-surface-elevated/30 rounded-full h-2">
                                                <div
                                                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                                  style={{ width: `${percentage}%` }}
                                                />
                                              </div>
                                              <span className="text-lg font-bold text-blue-300 min-w-[1.5rem] text-right">
                                                {count}
                                              </span>
                                            </div>
                                          </div>
                                        );
                                      })}
                                  </div>
                                </div>
                              )}
                              {Object.keys(tasksByRobotId).length > 0 && (
                                <div className="bg-surface-overlay/40 rounded-lg p-3 border border-border/30">
                                  <div className="flex items-center space-x-2 mb-3">
                                    <div className="w-5 h-5 bg-emerald-500/20 rounded border border-emerald-500/40 flex items-center justify-center">
                                      <span className="text-xs font-bold text-emerald-300">R</span>
                                    </div>
                                    <h4 className="text-sm font-semibold text-emerald-300">By Specific Robot</h4>
                                  </div>
                                  <div className="grid grid-cols-1 gap-2">
                                    {Object.entries(tasksByRobotId)
                                      .sort(([,a], [,b]) => b - a) // Sort by task count descending
                                      .map(([robotId, count]) => {
                                        const percentage = assignedTasks.length > 0 ? (count / assignedTasks.length) * 100 : 0;
                                        return (
                                          <div key={robotId} className="flex items-center justify-between p-2 bg-surface-overlay/30 rounded border border-border/20">
                                            <div className="flex items-center space-x-2">
                                              <span className="text-white text-sm font-medium px-2 py-1 bg-emerald-600/50 rounded">
                                                {robotId}
                                              </span>
                                              <span className="text-[var(--color-text-secondary)] text-xs">
                                                {percentage.toFixed(0)}%
                                              </span>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                              <div className="w-12 bg-surface-elevated/30 rounded-full h-2">
                                                <div
                                                  className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
                                                  style={{ width: `${percentage}%` }}
                                                />
                                              </div>
                                              <span className="text-lg font-bold text-emerald-300 min-w-[1.5rem] text-right">
                                                {count}
                                              </span>
                                            </div>
                                          </div>
                                        );
                                      })}
                                  </div>
                                </div>
                              )}
                            </div>
                        )}
                        </div>
                      </Card>

                      {/* Execution Readiness */}
                      <Card className="p-6 border-border">
                        <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                            <Wand2 className="w-4 h-4 text-purple-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Execution Readiness</h3>
                        </div>

                        {/* Top metrics in horizontal layout */}
                        <div className="grid grid-cols-3 gap-4">
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                               <div className={`text-2xl font-bold mb-1 ${
                                 allocationStatusValue === 'fully_allocated' ? 'text-blue-300' :
                                 allocationStatusValue === 'partially_allocated' ? 'text-yellow-300' :
                                 allocationStatusValue === 'unallocated' ? 'text-red-300' : 'text-[var(--color-text)]'
                               }`}>
                              {capitalize(allocationStatusValue?.replace('_', ' ') || 'Unknown')}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Plan Status
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className={`text-2xl font-bold mb-1 ${
                              executionReady ? 'text-blue-300' :
                              allocationStatusValue === 'fully_allocated' ? 'text-yellow-300' : 'text-red-300'
                            }`}>
                              {executionReady ? 'Yes' :
                               allocationStatusValue === 'fully_allocated' ? 'Robots Offline' : 'No'}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Execution Status
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-[var(--color-text)] mb-1">
                              {tasks.length > 0 ? Math.round((tasks.filter(t => !t.dependency_task_ids?.length).length / tasks.length) * 100) : 0}%
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Dependencies Met
                            </div>
                          </div>
                        </div>

                        {/* Show which robots are offline */}
                        {allocationStatusValue === 'fully_allocated' && offlineRobots.length > 0 && (
                          <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-5 h-5 bg-red-500/20 rounded border border-red-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-red-300">!</span>
                              </div>
                              <h4 className="text-sm font-semibold text-red-300">Offline Robots</h4>
                            </div>
                            <div className="text-red-200 text-sm mb-3">
                              The following robots are unreachable and must be online before execution:
                            </div>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                              {offlineRobots.map(robotId => (
                                <div key={robotId} className="flex items-center space-x-2 p-3 bg-red-500/5 border border-red-500/30 rounded-lg">
                                  <div className="w-8 h-8 bg-red-500/20 rounded border border-red-500/40 flex items-center justify-center">
                                    <span className="text-xs font-bold text-red-300">⚠</span>
                                  </div>
                                  <div>
                                    <div className="text-red-300 text-sm font-medium">{robotId}</div>
                                    <div className="text-red-400 text-xs">Unreachable</div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </Card>

                      {/* Goals Overview */}
                      <Card className="p-6 border-border">
                        <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                            <Target className="w-4 h-4 text-emerald-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Goals Overview</h3>
                        </div>

                        <div className="space-y-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                              <div className="text-2xl font-bold text-emerald-300 mb-1">
                                {plan?.goal_ids?.length || 0}
                              </div>
                              <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                                Total Goals
                              </div>
                            </div>
                            <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                              <div className="text-2xl font-bold text-emerald-300 mb-1">
                                {(() => {
                                  const goalTasks = tasks.filter(task => task.goal_id !== undefined)
                                  return goalTasks.length
                                })()}
                              </div>
                              <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                                Tasks with Goals
                              </div>
                            </div>
                          </div>

                          {plan?.goal_ids && plan.goal_ids.length > 0 && (
                            <div>
                              <div className="text-sm font-medium text-[var(--color-text)] mb-3">Goal IDs:</div>
                              <div className="flex flex-wrap gap-2">
                                {plan.goal_ids.map(goalId => {
                                  const taskCount = tasks.filter((task: any) => task.goal_id === goalId).length
                                  return (
                                    <div key={goalId} className="bg-surface-overlay/50 px-3 py-2 rounded-lg border border-border">
                                      <div className="text-sm font-medium text-emerald-300">#{goalId}</div>
                                      <div className="text-xs text-[var(--color-text-secondary)]">{taskCount} tasks</div>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      </Card>

                      {/* Research Metrics */}
                      <Card className="p-6 border-border">
                        <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-orange-500/20 rounded-lg flex items-center justify-center">
                            <Link className="w-4 h-4 text-orange-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Research Metrics</h3>
                        </div>

                        {/* Top metrics in horizontal layout */}
                        <div className="grid grid-cols-3 gap-4">
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-orange-300 mb-1">
                              {Math.max(...tasks.map(t => t.dependency_task_ids?.length || 0), 0)}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Max Dependencies
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-orange-300 mb-1">
                              {tasks.length > 0 ? (tasks.reduce((sum, t) => sum + (t.dependency_task_ids?.length || 0), 0) / tasks.length).toFixed(1) : 0}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Avg Dependencies
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border text-center">
                            <div className="text-2xl font-bold text-orange-300 mb-1">
                              {tasks.filter(t => !t.dependency_task_ids?.length).length}
                            </div>
                            <div className="text-[var(--color-text-secondary)] text-xs font-medium">
                              Parallel Tasks
                            </div>
                          </div>
                        </div>
                      </Card>

                    </div>
                  );
                })()}
                </div>
              );
            })()}

            {activeTab === 'tasks' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                      <GitBranch className="w-5 h-5 text-emerald-400" />
                    </div>
                    <h3 className="text-xl font-bold text-white">Task Details</h3>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="secondary" size="sm" onClick={openCreateTask}>
                      <Plus className="w-4 h-4" />
                      Add Task
                    </Button>
                    <button
                      onClick={() => setTaskView('vertical')}
                      className={cn(
                        'px-3 py-1 text-sm rounded transition-colors',
                        taskView === 'vertical'
                          ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      )}
                    >
                      List View
                    </button>
                    <button
                      onClick={() => setTaskView('dag')}
                      className={cn(
                        'px-3 py-1 text-sm rounded transition-colors',
                        taskView === 'dag'
                          ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      )}
                    >
                      DAG View
                    </button>
                    <button
                      onClick={downloadDAG}
                      className="px-3 py-1 text-sm rounded transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-text)] border border-border hover:border-border-strong"
                    >
                      📥 Download DAG (SVG)
                    </button>
                  </div>
                </div>

                {taskView === 'vertical' && (
                  <VerticalTaskList tasks={plan.tasks || []} onEdit={openEditTask} onDelete={requestDeleteTask} />
                )}
                {taskView === 'dag' && <DAGVisualization tasks={plan.tasks || []} />}
              </div>
            )}

            {activeTab === 'goals' && (
              <div className="space-y-6">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                    <Target className="w-5 h-5 text-emerald-400" />
                  </div>
                  <h3 className="text-xl font-bold text-white">Goals</h3>
                </div>

                {plan?.goal_ids && plan.goal_ids.length > 0 ? (
                  <div className="grid gap-4">
                    {plan.goal_ids.map(goalId => {
                      const goal = (goals as any[]).find((g: any) => g.goal_id === goalId)
                      const goalTasks = (plan.tasks || []).filter((task: any) => task.goal_id === goalId)

                      return (
                        <Card key={goalId} className="p-4 border-border">
                          <div className="flex items-start space-x-4">
                            <div className="w-12 h-12 bg-emerald-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                              <Target className="w-6 h-6 text-emerald-400" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-3 mb-2">
                                <h4 className="text-lg font-semibold text-white">Goal #{goalId}</h4>
                                <span className="px-2 py-1 bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 rounded text-sm">
                                  {goalTasks.length} tasks
                                </span>
                              </div>
                              <p className="text-[var(--color-text)] mb-3">{goal?.description || 'Goal description not available'}</p>

                              {goalTasks.length > 0 && (
                                <div>
                                  <div className="text-sm font-medium text-[var(--color-text-secondary)] mb-2">Tasks in this plan:</div>
                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                    {goalTasks.map(task => (
                                      <div key={task.task_id} className="bg-surface-overlay/50 px-3 py-2 rounded border border-border">
                                        <div className="flex items-center justify-between">
                                          <span className="text-sm font-medium text-[var(--color-text)]">Task #{task.task_id}</span>
                                          <span className={cn(
                                            'px-2 py-0.5 rounded text-xs',
                                            task.status === 'completed' ? 'bg-green-500/20 text-green-300' :
                                            task.status === 'in_progress' ? 'bg-blue-500/20 text-blue-300' :
                                            task.status === 'failed' ? 'bg-red-500/20 text-red-300' :
                                            'bg-gray-500/20 text-gray-300'
                                          )}>
                                            {task.status}
                                          </span>
                                        </div>
                                        <div className="text-xs text-[var(--color-text-secondary)] mt-1 truncate">{task.description}</div>
                                        {task.robot_id && (
                                          <div className="text-xs text-[var(--color-text-muted)] mt-1">Robot: {task.robot_id}</div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </Card>
                      )
                    })}
                  </div>
                ) : (
                  <EmptyState
                    icon={<Target className="w-8 h-8" />}
                    title="No goals assigned"
                    description="This plan doesn't have any goals assigned to it."
                  />
                )}
              </div>
            )}

            {activeTab === 'prompts' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                      <Wand2 className="w-5 h-5 text-blue-400" />
                    </div>
                    <h3 className="text-xl font-bold text-white">Prompts</h3>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setPromptsView('planning')}
                      className={cn(
                        'px-3 py-1 text-sm rounded transition-colors',
                        promptsView === 'planning'
                          ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      )}
                    >
                      Planning
                    </button>
                    <button
                      onClick={() => setPromptsView('allocation')}
                      className={cn(
                        'px-3 py-1 text-sm rounded transition-colors',
                        promptsView === 'allocation'
                          ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      )}
                    >
                      Allocation
                    </button>
                  </div>
                </div>

                {promptsView === 'planning' && (
                  <Card className="p-6 border-border">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                          <GitBranch className="w-4 h-4 text-blue-400" />
                        </div>
                        <h4 className="text-lg font-bold text-white">Planning Prompts</h4>
                      </div>
                      <button
                        onClick={downloadPlanningPrompts}
                        className="px-3 py-1 text-sm rounded transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-text)] border border-border hover:border-border-strong"
                      >
                        📥 Download (.zip)
                      </button>
                    </div>
                  {plan.planning_prompts ? (
                    <div className="space-y-4">
                      {typeof plan.planning_prompts === 'object' && plan.planning_prompts.system && plan.planning_prompts.user ? (
                        <div className="space-y-4">
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-6 h-6 bg-blue-500/20 rounded border border-blue-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-blue-300">S</span>
                              </div>
                              <h4 className="text-sm font-semibold text-blue-300">System Prompt</h4>
                            </div>
                            <div className="bg-surface/80 p-4 rounded border border-border">
                              <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                                {plan.planning_prompts.system}
                              </pre>
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-6 h-6 bg-green-500/20 rounded border border-green-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-green-300">U</span>
                              </div>
                              <h4 className="text-sm font-semibold text-green-300">User Prompt</h4>
                            </div>
                            <div className="bg-surface/80 p-4 rounded border border-border">
                              <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                                {plan.planning_prompts.user}
                              </pre>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border">
                          <div className="bg-surface/80 p-4 rounded border border-border">
                            <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                              {JSON.stringify(plan.planning_prompts, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="bg-surface-overlay/40 p-4 rounded-lg border border-border/30">
                      <div className="flex items-center space-x-3 text-[var(--color-text-secondary)]">
                        <div className="w-8 h-8 bg-surface-elevated/30 rounded-lg flex items-center justify-center">
                          <GitBranch className="w-4 h-4" />
                        </div>
                        <span>No planning prompts available for this plan.</span>
                      </div>
                    </div>
                  )}
                  </Card>
                )}

                {promptsView === 'allocation' && (
                  <Card className="p-6 border-border">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                          <Wand2 className="w-4 h-4 text-purple-400" />
                        </div>
                        <h4 className="text-lg font-bold text-white">Allocation Prompts</h4>
                      </div>
                      <button
                        onClick={downloadAllocationPrompts}
                        className="px-3 py-1 text-sm rounded transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-text)] border border-border hover:border-border-strong"
                      >
                        📥 Download (.zip)
                      </button>
                    </div>
                  {plan.allocation_prompts ? (
                    <div className="space-y-4">
                      {typeof plan.allocation_prompts === 'object' && plan.allocation_prompts.system && plan.allocation_prompts.user ? (
                        <div className="space-y-4">
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-6 h-6 bg-purple-500/20 rounded border border-purple-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-purple-300">S</span>
                              </div>
                              <h4 className="text-sm font-semibold text-purple-300">System Prompt</h4>
                            </div>
                            <div className="bg-surface/80 p-4 rounded border border-border">
                              <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                                {plan.allocation_prompts.system}
                              </pre>
                            </div>
                          </div>
                          <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-6 h-6 bg-green-500/20 rounded border border-green-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-green-300">U</span>
                              </div>
                              <h4 className="text-sm font-semibold text-green-300">User Prompt</h4>
                            </div>
                            <div className="bg-surface/80 p-4 rounded border border-border">
                              <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                                {plan.allocation_prompts.user}
                              </pre>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border">
                          <div className="bg-surface/80 p-4 rounded border border-border">
                            <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                              {JSON.stringify(plan.allocation_prompts, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="bg-surface-overlay/40 p-4 rounded-lg border border-border/30">
                      <div className="flex items-center space-x-3 text-[var(--color-text-secondary)]">
                        <div className="w-8 h-8 bg-surface-elevated/30 rounded-lg flex items-center justify-center">
                          <Wand2 className="w-4 h-4" />
                        </div>
                        <span>No allocation prompts available for this plan. Plan may not have been allocated yet.</span>
                      </div>
                    </div>
                  )}
                  </Card>
                )}
              </div>
            )}

            {activeTab === 'artifacts' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                      <Link className="w-5 h-5 text-blue-400" />
                    </div>
                    <h3 className="text-xl font-bold text-white">Artifacts</h3>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setArtifactsView('planning')}
                      className={cn(
                        'px-3 py-1 text-sm rounded transition-colors',
                        artifactsView === 'planning'
                          ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      )}
                    >
                      Planning
                    </button>
                    <button
                      onClick={() => setArtifactsView('allocation')}
                      className={cn(
                        'px-3 py-1 text-sm rounded transition-colors',
                        artifactsView === 'allocation'
                          ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      )}
                    >
                      Allocation
                    </button>
                  </div>
                </div>

                {artifactsView === 'planning' && (
                  <Card className="p-6 border-border">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                          <Link className="w-4 h-4 text-blue-400" />
                        </div>
                        <h4 className="text-lg font-bold text-white">Planning Artifacts</h4>
                      </div>
                      <button
                        onClick={downloadPlanningArtifacts}
                        className="px-3 py-1 text-sm rounded transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-text)] border border-border hover:border-border-strong"
                      >
                        📥 Download (.json)
                      </button>
                    </div>
                  {plan.planning_artifacts ? (
                    <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border">
                      <div className="bg-surface/80 p-4 rounded border border-border">
                        <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                          {JSON.stringify(plan.planning_artifacts, null, 2)}
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-surface-overlay/40 p-4 rounded-lg border border-border/30">
                      <div className="flex items-center space-x-3 text-[var(--color-text-secondary)]">
                        <div className="w-8 h-8 bg-surface-elevated/30 rounded-lg flex items-center justify-center">
                          <Link className="w-4 h-4" />
                        </div>
                        <span>No planning artifacts available for this plan.</span>
                      </div>
                    </div>
                  )}
                  </Card>
                )}

                {artifactsView === 'allocation' && (
                  <Card className="p-6 border-border">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                          <Link className="w-4 h-4 text-purple-400" />
                        </div>
                        <h4 className="text-lg font-bold text-white">Allocation Artifacts</h4>
                      </div>
                      <button
                        onClick={downloadAllocationArtifacts}
                        className="px-3 py-1 text-sm rounded transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-text)] border border-border hover:border-border-strong"
                      >
                        📥 Download (.json)
                      </button>
                    </div>
                  {plan.allocation_artifacts ? (
                    <div className="bg-surface-overlay/50 p-4 rounded-lg border border-border">
                      <div className="bg-surface/80 p-4 rounded border border-border">
                        <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                          {JSON.stringify(plan.allocation_artifacts, null, 2)}
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-surface-overlay/40 p-4 rounded-lg border border-border/30">
                      <div className="flex items-center space-x-3 text-[var(--color-text-secondary)]">
                        <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                          <Link className="w-4 h-4" />
                        </div>
                        <span>No allocation artifacts available for this plan. Plan may not have been allocated yet.</span>
                      </div>
                    </div>
                  )}
                  </Card>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>

      {/* Method Detail Modal */}
      <MethodDetailModal
        methodId={selectedMethod?.id || null}
        methodType={selectedMethod?.type || undefined}
        isOpen={!!selectedMethod}
        onClose={() => {
          setSelectedMethod(null)
          navigate(`/plans/${planId}`, { replace: true })
        }}
      />

      {/* Allocate Plan Modal */}
      {allocatePlanId && (
        <Modal
          isOpen={!!allocatePlanId}
          onClose={() => setAllocatePlanId(null)}
          title={`Allocate Plan #${allocatePlanId}`}
          size="wide"
        >
          <AllocatePlanForm
            robots={robots || []}
            onSubmit={(allocationStrategy) => {
              allocateMutation.mutate({ planId: allocatePlanId, allocationStrategy })
            }}
            onCancel={() => setAllocatePlanId(null)}
            isLoading={allocateMutation.isPending}
          />
        </Modal>
      )}

      {/* Task Editor Modal */}
      <Modal
        isOpen={isTaskEditorOpen}
        onClose={() => setIsTaskEditorOpen(false)}
        title={taskEditorMode === 'create' ? 'Add Task' : `Edit Task #${editingTaskId ?? ''}`}
        size="wide"
      >
        <form onSubmit={submitTaskEditor} className="space-y-6 max-h-[75vh] overflow-y-auto">
          <div className="space-y-2">
            <div className="text-sm font-medium text-white">Description</div>
            <textarea
              value={taskFormDescription}
              onChange={(e) => setTaskFormDescription(e.target.value)}
              className="w-full min-h-[96px] bg-surface-overlay/60 border border-border rounded-lg px-3 py-2 text-white placeholder:text-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-cyber-500/40"
              placeholder="Describe what this task should do…"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="text-sm font-medium text-white">Goal</div>
              <select
                value={taskFormGoalId === '' ? '' : String(taskFormGoalId)}
                onChange={(e) => setTaskFormGoalId(e.target.value ? Number(e.target.value) : '')}
                className="w-full bg-surface-overlay/60 border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyber-500/40"
              >
                <option value="">Select a goal…</option>
                {((plan?.goal_ids && plan.goal_ids.length > 0 ? plan.goal_ids : (goals as any[]).map((g: any) => g.goal_id)) as number[]).map(
                  (goalId) => {
                    const goal = (goals as any[]).find((g: any) => g.goal_id === goalId)
                    return (
                      <option key={goalId} value={goalId}>
                        Goal #{goalId}{goal?.description ? ` — ${goal.description}` : ''}
                      </option>
                    )
                  }
                )}
              </select>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium text-white">Robot (optional)</div>
              <select
                value={taskFormRobotId}
                onChange={(e) => setTaskFormRobotId(e.target.value)}
                className="w-full bg-surface-overlay/60 border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyber-500/40"
              >
                <option value="">Unassigned</option>
                {(robots as any[] | undefined)?.map((r: any) => (
                  <option key={r.robot_id} value={r.robot_id}>
                    {r.robot_id} ({r.robot_type})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium text-white">Dependencies</div>
              <div className="text-xs text-[var(--color-text-muted)]">Select tasks that must finish before this one</div>
            </div>
            <div className="bg-surface/60 border border-border/50 rounded-lg p-3 max-h-48 overflow-y-auto space-y-2">
              {(plan?.tasks || [])
                .filter((t: any) => t.task_id && t.task_id !== editingTaskId)
                .map((t: any) => {
                  const checked = taskFormDependencyIds.includes(t.task_id)
                  return (
                    <label key={t.task_id} className="flex items-start gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {
                          setTaskFormDependencyIds((prev) =>
                            checked ? prev.filter((id) => id !== t.task_id) : [...prev, t.task_id]
                          )
                        }}
                        className="mt-1"
                      />
                      <div className="min-w-0">
                        <div className="text-sm text-white font-medium">Task #{t.task_id}</div>
                        <div className="text-xs text-[var(--color-text-muted)] truncate">{t.description}</div>
                      </div>
                    </label>
                  )
                })}
              {(plan?.tasks || []).filter((t: any) => t.task_id && t.task_id !== editingTaskId).length === 0 && (
                <div className="text-sm text-[var(--color-text-muted)]">No other tasks available.</div>
              )}
            </div>
          </div>

          <div className="flex items-center justify-end gap-2 pt-2">
            <Button variant="secondary" type="button" onClick={() => setIsTaskEditorOpen(false)}>
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={createTaskMutation.isPending || updateTaskMutation.isPending}
            >
              {taskEditorMode === 'create' ? 'Create Task' : 'Save Changes'}
            </Button>
          </div>
        </form>
      </Modal>

      {/* Delete Task Confirmation */}
      <Modal
        isOpen={deleteTaskId != null}
        onClose={() => setDeleteTaskId(null)}
        title={deleteTaskId != null ? `Delete Task #${deleteTaskId}` : 'Delete Task'}
        size="lg"
      >
        <div className="space-y-4">
          <div className="text-[var(--color-text)]">
            This will delete the task and automatically remove it from any other tasks’ dependency lists (dependents are not deleted).
          </div>
          <div className="flex items-center justify-end gap-2">
            <Button variant="secondary" onClick={() => setDeleteTaskId(null)} disabled={deleteTaskMutation.isPending}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => deleteTaskId != null && deleteTaskMutation.mutate(deleteTaskId)}
              disabled={deleteTaskMutation.isPending || deleteTaskId == null}
            >
              <Trash2 className="w-4 h-4" />
              Delete
            </Button>
          </div>
        </div>
      </Modal>
    </>
  )
}

// =============================================================================
// Allocation Components (copied from Plans.tsx)
// =============================================================================

interface AllocatePlanFormProps {
  robots: Robot[]
  onSubmit: (allocationStrategy: string) => void
  onCancel: () => void
  isLoading: boolean
}

function MethodSelectionCard({
  method,
  isSelected,
  onClick
}: {
  method: MethodSummary
  isSelected: boolean
  onClick: () => void
}) {
  return (
    <Card
      className={cn(
        'cursor-pointer hover:border-border transition-all p-3',
        isSelected && 'ring-2 ring-cyber-500/50 shadow-lg shadow-cyber-500/20 border-cyber-500'
      )}
      onClick={onClick}
    >
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-white text-sm">{method.name}</h3>
          <div className="flex items-center gap-2">
            <span className={cn(
              'px-2 py-0.5 rounded text-xs font-medium border',
              method.method_type === 'foundation model' && 'bg-blue-500/10 border-blue-500/30 text-blue-400',
              method.method_type === 'hybrid' && 'bg-purple-500/10 border-purple-500/30 text-purple-400',
              method.method_type === 'algorithmic' && 'bg-orange-500/10 border-orange-500/30 text-orange-400',
              method.method_type === 'manual' && 'bg-surface-elevated/10 border-border text-[var(--color-text-secondary)]'
            )}>
              {method.method_type}
            </span>
            {isSelected && (
              <CheckCircle className="w-4 h-4 text-cyber-400" />
            )}
          </div>
        </div>
        <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed line-clamp-2">{method.description}</p>
      </div>
    </Card>
  )
}

function AllocatePlanForm({ robots, onSubmit, onCancel, isLoading }: AllocatePlanFormProps) {
  const [selectedAllocator, setSelectedAllocator] = useState<string | null>(null)
  const [allocatorFilter, setAllocatorFilter] = useState<string>('all')

  // Load available allocators dynamically
  const { data: allocators = [], isLoading: allocatorsLoading, error: allocatorsError } = useQuery({
    queryKey: ['allocators'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'allocator')),
  })

  // Allocation options (only real allocators, no manual option)
  const allocationOptions = [...allocators]

  // Filter allocators by method type
  const filteredAllocators = allocationOptions.filter(a =>
    allocatorFilter === 'all' || a.method_type === allocatorFilter
  )

  // Get unique method types for filters
  const allocatorMethodTypes = ['all', ...new Set(allocationOptions.map(a => a.method_type))]

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedAllocator) return
    onSubmit(selectedAllocator)
  }

  // If still loading or error, show loading state
  if (allocatorsLoading) {
    return (
      <div className="p-8 text-center text-[var(--color-text-secondary)]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-border mx-auto mb-4"></div>
        Loading allocation methods...
      </div>
    )
  }

  if (allocatorsError) {
    return (
      <div className="p-8 text-center text-red-400">
        Error loading allocation methods: {allocatorsError.message}
      </div>
    )
  }

  return (
    <div className="space-y-6 max-h-[75vh] overflow-y-auto">
      {allocators.length === 0 ? (
        <div className="text-center py-8 text-[var(--color-text-secondary)]">
          No allocation methods available. Please check your configuration.
        </div>
      ) : (
        <>
          {/* Allocation Methods - Same design as CreatePlanForm */}
          <div>
        <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-3">
          <span className="w-3 h-3 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full"></span>
          Choose Allocation Method
        </h3>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-medium text-white flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
              Allocation Methods
            </h4>
            {/* Method Type Filter - Below header */}
            <div className="flex gap-1 flex-wrap">
              {allocatorMethodTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => setAllocatorFilter(type)}
                  className={cn(
                    'px-3 py-1 text-xs rounded-full border transition-all',
                    allocatorFilter === type
                      ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                      : 'border-border text-[var(--color-text-secondary)] hover:border-border-strong'
                  )}
                >
                  {type === 'all' ? 'All' : type}
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {filteredAllocators.map((allocator) => (
              <MethodSelectionCard
                key={allocator.type}
                method={allocator}
                isSelected={selectedAllocator === allocator.type}
                onClick={() => setSelectedAllocator(allocator.type)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Available Robots */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-3">
          <span className="w-3 h-3 bg-gradient-to-r from-violet-400 to-pink-400 rounded-full"></span>
          Available Robots
        </h3>
        <div className="mb-4">
          <div className="text-sm text-[var(--color-text-secondary)]">
            {robots.filter(r => r.status === 'running' || r.status === 'registered').length} of {robots.length} robots available
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-64 overflow-y-auto">
          {robots.map((robot) => (
            <Card
              key={robot.robot_id}
              className={cn(
                'p-3 transition-all',
                robot.status === 'running' || robot.status === 'registered'
                  ? 'bg-emerald-500/10 border-emerald-500/30'
                  : 'bg-surface-overlay/50 border-border'
              )}
            >
              <div className="flex items-center gap-3">
                <div className={cn(
                  'w-3 h-3 rounded-full flex-shrink-0',
                  robot.status === 'running' || robot.status === 'registered'
                    ? 'bg-emerald-400 shadow-lg shadow-emerald-400/50'
                    : 'bg-surface-elevated'
                )} />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-white text-sm truncate">
                    {robot.robot_id}
                  </div>
                  <div className="text-xs text-[var(--color-text-secondary)]">
                    {robot.robot_type}
                  </div>
                </div>
                <div className={cn(
                  'text-xs px-2 py-1 rounded-full',
                  robot.status === 'running' || robot.status === 'registered'
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-surface-elevated text-[var(--color-text-secondary)]'
                )}>
                  {robot.status === 'running' || robot.status === 'registered' ? 'Available' : 'Offline'}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end gap-3 pt-4 border-t border-border">
        <Button variant="secondary" onClick={onCancel}>
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          disabled={!selectedAllocator || isLoading}
          className="flex items-center gap-2"
        >
          {isLoading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              Allocating...
            </>
          ) : (
            <>
              <Users className="w-4 h-4" />
              Allocate Robots
            </>
          )}
        </Button>
      </div>
        </>
      )}
    </div>
  )
}