import React, { useState, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import {
  ArrowLeft, Wand2, GitBranch, Link, Bot, Target, Play, Edit, Check, X, Zap, Download, Loader2
} from 'lucide-react'
// Graphviz is imported dynamically below

import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { EmptyState } from '../components/common/EmptyState'
import { MethodDetailModal } from './Planners'
import { plansApi, robotsApi, goalsApi, methodsApi, useRealtimeUpdates } from '../lib/api'
import { cn, capitalize, getPlanningStrategyName, getAllocationStrategyName, getPlanningMethodId, getAllocationMethodId, setMethodData } from '../lib/utils'
import JSZip from 'jszip'

// =============================================================================
// Task List Component
// =============================================================================

function VerticalTaskList({ tasks }: { tasks: any[] }) {
  return (
    <div className="space-y-4">
      {tasks.map((task, index) => (
        <Card key={task.task_id || index} className="p-6 hover:bg-slate-800/50 transition-colors">
          <div className="space-y-4">
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <div className="flex items-center space-x-3">
                  <span className="text-xl font-bold text-white">Task {task.task_id}</span>
                  <div className="flex items-center space-x-2">
                    <span className="px-3 py-1 bg-blue-500/20 border border-blue-500/40 text-blue-300 text-sm font-semibold rounded-lg">
                      {task.robot_type}
                    </span>
                    {task.robot_id && (
                      <span className="px-3 py-1 bg-green-500/20 border border-green-500/40 text-green-300 text-sm font-semibold rounded-lg">
                        {task.robot_id}
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-slate-300 text-base leading-relaxed">{task.description}</div>
              </div>
              <div className="flex flex-col items-end space-y-2">
                {task.status && (
                  <span className="px-3 py-1 bg-slate-500/20 border border-slate-500/40 text-slate-300 text-sm font-medium rounded-lg">
                    {task.status}
                  </span>
                )}
              </div>
            </div>

            {task.dependency_task_ids && task.dependency_task_ids.length > 0 && (
              <div className="bg-slate-800/30 p-3 rounded-lg border border-slate-600/50">
                <div className="flex items-center space-x-2 mb-2">
                  <Link className="w-4 h-4 text-slate-400" />
                  <span className="text-sm font-medium text-slate-300">Dependencies</span>
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
// Graphviz DAG Visualization Component
// =============================================================================

function DAGVisualization({ tasks }: { tasks: any[] }) {
  console.log('DAGVisualization: Called with tasks:', tasks?.length)

  const [svgContent, setSvgContent] = React.useState<string>('')

  // Debug svgContent changes
  React.useEffect(() => {
    console.log('svgContent changed, length:', svgContent?.length || 0)
  }, [svgContent])
  const [isGenerating, setIsGenerating] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [zoom, setZoom] = React.useState(0.6)
  const [pan, setPan] = React.useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = React.useState(false)
  const [lastMousePos, setLastMousePos] = React.useState({ x: 0, y: 0 })
  const containerRef = React.useRef<HTMLDivElement>(null)

  // Download functions defined below after plan data is available

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'r' || e.key === 'R') {
        resetView()
      } else if (e.key === '+' || e.key === '=') {
        setZoom(prev => Math.min(3, prev * 1.2))
      } else if (e.key === '-') {
        setZoom(prev => Math.max(0.1, prev * 0.8))
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Mouse event handlers for zoom and pan
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) { // Left click only
      setIsDragging(true)
      setLastMousePos({ x: e.clientX, y: e.clientY })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      const deltaX = e.clientX - lastMousePos.x
      const deltaY = e.clientY - lastMousePos.y
      setPan(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }))
      setLastMousePos({ x: e.clientX, y: e.clientY })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1
    const newZoom = Math.max(0.1, Math.min(3, zoom * zoomFactor))
    setZoom(newZoom)
  }

  const resetView = () => {
    setZoom(0.2)
    setPan({ x: 0, y: 0 })
  }

  // Touch event handlers for mobile
  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      setIsDragging(true)
      setLastMousePos({ x: e.touches[0].clientX, y: e.touches[0].clientY })
    }
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    if (isDragging && e.touches.length === 1) {
      e.preventDefault()
      const deltaX = e.touches[0].clientX - lastMousePos.x
      const deltaY = e.touches[0].clientY - lastMousePos.y
      setPan(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }))
      setLastMousePos({ x: e.touches[0].clientX, y: e.touches[0].clientY })
    }
  }

  const handleTouchEnd = () => {
    setIsDragging(false)
  }

  React.useEffect(() => {
    console.log('DAGVisualization: Tasks changed, regenerating DAG:', tasks?.length)

    if (!tasks || tasks.length === 0) {
      console.log('DAGVisualization: No tasks')
      setSvgContent('')
      setError(null)
      setIsGenerating(false)
      return
    }

    setIsGenerating(true)
    setError(null)

    // Generate Graphviz DOT format with beautiful styling
    const generateDot = (tasks: any[]) => {
      let dot = `digraph G {
  rankdir=LR;
  bgcolor="#ffffff";
  node [shape=plaintext, fontname="Arial"];
  edge [color="#334155",penwidth=5.0, arrowhead=vee, arrowsize=4.0, headclip=true, tailclip=true];
  graph [splines=spline, nodesep=1.2, ranksep=1.4];
`

      // Add nodes with beautiful card-like HTML styling
      tasks.forEach(task => {
        // Status-based colors for light background with better contrast
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
            statusColors = { border: '#1d4ed8', bg: '#dbeafe', text: '#1e3a8a', headerBg: '#bfdbfe' }
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

        // Create beautiful HTML table label
        const taskId = task.task_id.toString()
        const description = task.description.replace(/"/g, '\\"').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        const robotType = task.robot_type || 'unknown'
        const robotId = task.robot_id || 'unassigned'

        const htmlLabel = `<TABLE BORDER="3" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" BGCOLOR="${statusColors.bg}" COLOR="${statusColors.border}" STYLE="ROUNDED">
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="${statusColors.headerBg}" CELLPADDING="18">
      <FONT COLOR="#1d4ed8" FACE="Arial" POINT-SIZE="50">
        <B>Task ${taskId}</B>
      </FONT>
    </TD>
  </TR>

  <TR>
    <TD ALIGN="CENTER" CELLPADDING="20">
      <FONT COLOR="${statusColors.text}" FACE="Arial" POINT-SIZE="46">
        <b>${description}</b>
      </FONT>
    </TD>
  </TR>

  <TR>
    <TD ALIGN="CENTER" CELLPADDING="16">
      <TABLE BORDER="0" CELLSPACING="12">
        <TR>
          <TD BGCOLOR="#7c3aed" CELLPADDING="14">
            <FONT COLOR="#ffffff" FACE="Arial" POINT-SIZE="40">
              <B>${robotType}</B>
            </FONT>
          </TD>
          <TD BGCOLOR="#0d9488" CELLPADDING="14">
            <FONT COLOR="#ffffff" FACE="Arial" POINT-SIZE="40">
              <B>${robotId}</B>
            </FONT>
          </TD>
        </TR>
      </TABLE>
    </TD>
  </TR>
</TABLE>`

        dot += `  ${task.task_id} [label=<${htmlLabel}>];\n`
      })

      // Add edges with subtle professional styling
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
    console.log('Graphviz DOT:', dotSource)

    // Generate SVG using Graphviz with correct async loading pattern
    try {
      console.log('DAGVisualization: Importing Graphviz...')
      import('@hpcc-js/wasm-graphviz').then(async ({ Graphviz }) => {
        console.log('DAGVisualization: Graphviz imported, loading WASM...')
        try {
          const graphviz = await Graphviz.load()
          console.log('DAGVisualization: Graphviz WASM loaded successfully')
          console.log('DAGVisualization: Graphviz version:', graphviz.version())

          console.log('DAGVisualization: Rendering DOT to SVG...')
          const svg = graphviz.dot(dotSource, 'svg')
          console.log('DAGVisualization: SVG generated successfully, length:', svg.length)
          console.log('DAGVisualization: First 200 chars of SVG:', svg.substring(0, 200))
          console.log('DAGVisualization: DOT source preview:', dotSource.substring(0, 300) + '...')

          if (svg && svg.length > 100 && svg.includes('<svg')) { // Basic validation
            setSvgContent(svg)
            console.log('DAGVisualization: SVG content set successfully')
          } else {
            console.error('DAGVisualization: Generated SVG is invalid (too short or missing SVG tag):', svg.substring(0, 500))
            setError('Generated SVG is invalid')
          }
          setIsGenerating(false)
          console.log('DAGVisualization: svgContent updated in parent state')
        } catch (loadError: any) {
          console.error('DAGVisualization: Graphviz load/render error:', loadError)
          setError(`Graphviz failed: ${loadError.message}`)
          setIsGenerating(false)
        }
      }).catch((importError: any) => {
        console.error('DAGVisualization: Graphviz import failed:', importError)
        setError(`Graphviz import failed: ${importError.message}`)
        setIsGenerating(false)
      })
    } catch (err: any) {
      console.error('DAGVisualization: Setup error:', err)
      setError(`Graphviz setup error: ${err.message}`)
      setIsGenerating(false)
    }
  }, [tasks]) // Re-run when tasks change for live updates

  if (!tasks || tasks.length === 0) {
    return (
      <div className="text-center py-8 text-slate-400 bg-slate-800/50 rounded-lg">
        No tasks to visualize in DAG
      </div>
    )
  }

  if (isGenerating) {
    return (
      <div className="text-center py-8 text-slate-400 bg-slate-800/50 rounded-lg">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-slate-400 mx-auto mb-2"></div>
        Generating DAG visualization...
      </div>
    )
  }

  if (error) {
    // Fallback: Simple text-based DAG representation
    return (
      <div className="space-y-4">
        <div className="text-sm text-red-400 text-center">
          Graphviz failed - showing text representation
        </div>
        <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 overflow-auto max-h-96">
          <div className="font-mono text-sm text-slate-300 space-y-2">
            <div className="text-slate-400 mb-4">Task Dependencies:</div>
            {tasks.map(task => (
              <div key={task.task_id} className="flex items-start space-x-4">
                <div className="text-blue-400 font-bold min-w-[3rem]">T{task.task_id}</div>
                <div className="flex-1">
                  <div className="text-white">{task.description}</div>
                  {task.dependency_task_ids && task.dependency_task_ids.length > 0 && (
                    <div className="text-slate-400 text-xs mt-1">
                      Depends on: {task.dependency_task_ids.join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="text-xs text-slate-500 text-center">
          Error: {error}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-600">
          Professional DAG Visualization • Interactive zoom & pan • Live status updates
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-600">Zoom: {Math.round(zoom * 100)}%</span>
          <Button
            variant="secondary"
            size="sm"
            onClick={resetView}
            className="text-xs"
          >
            Reset View
          </Button>
        </div>
      </div>

      <div
        ref={containerRef}
        className="bg-white border border-slate-300 rounded-lg overflow-hidden shadow-lg"
        style={{ height: '600px', cursor: isDragging ? 'grabbing' : 'grab' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <div
          key={`dag-${Date.now()}-${tasks?.length}-${tasks?.map(t => t.status).join('-')}`}
          className="w-full h-full flex items-center justify-center p-4"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.1s ease-out'
          }}
          dangerouslySetInnerHTML={{ __html: svgContent }}
        />
      </div>

      <div className="text-xs text-slate-200 text-center space-y-1">
        <div>🔗 Arrows show dependencies (A → B means B depends on A)</div>
        <div>🖱️ Drag to pan • Scroll to zoom • ⌨️ R: Reset • +/-: Zoom</div>
        <div>💎 Card-style nodes • 🟣 Purple pills: Robot type • 🟢 Teal pills: Robot ID</div>
        <div>✅ Green: Completed • 🔵 Blue: Running • 🔴 Red: Failed • ⚪ Gray: Pending</div>
        <div>
          <Download className="w-4 h-4 mr-2 inline" />
          Download full plan as organized zip with prompts/artifacts/dag folders
        </div>
      </div>
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
  const [activeTab, setActiveTab] = useState<'overview' | 'tasks' | 'prompts' | 'artifacts' | 'goals'>('overview')
  const [taskView, setTaskView] = useState<'vertical' | 'dag'>('vertical')
  const [promptsView, setPromptsView] = useState<'planning' | 'allocation'>('planning')
  const [artifactsView, setArtifactsView] = useState<'planning' | 'allocation'>('planning')
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [justUpdated, setJustUpdated] = useState(false)

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

  // Check if plan is executable (has allocated tasks and robots are available)
  const isPlanExecutable = plan && allocationStatus && robotHealth && robots ? (() => {
    if (allocationStatus.status !== 'fully_allocated') return false

    // Get all robots assigned to tasks in this plan
    const planTasks = plan.tasks || []
    const assignedTasks = planTasks.filter(t => t.robot_id)
    const assignedRobotIds = [...new Set(assignedTasks.map(t => t.robot_id).filter(Boolean))]

    // Check if all assigned robots are reachable
    return assignedRobotIds.every(robotId => {
      const robot = robots.find(r => r.robot_id === robotId)
      const health = robotHealth[robotId]
      return robot && health?.reachable === true
    })
  })() : false

  // Execution mutation
  const startMutation = useMutation({
    mutationFn: (planId: number) => plansApi.start(planId),
    onSuccess: () => {
      // Navigate to execution page
      navigate(`/execution/${planId}`)
    },
  })

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
      <div className="min-h-screen bg-slate-900 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center py-12">
            <div className="text-slate-400">Loading plan details...</div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-900 p-6">
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
      <div className="min-h-screen bg-slate-900 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center py-12">
            <div className="text-slate-400">Plan not found</div>
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
              statusColors = { border: '#1d4ed8', bg: '#dbeafe', text: '#1e3a8a', headerBg: '#bfdbfe' }
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
                statusColors = { border: '#1d4ed8', bg: '#dbeafe', text: '#1e3a8a', headerBg: '#bfdbfe' }
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
      <div className="min-h-screen bg-slate-900 p-6">
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

          {/* Execute Button */}
          {plan && (
            <Button
              onClick={() => startMutation.mutate(plan.plan_id)}
              disabled={startMutation.isPending || !isPlanExecutable || plan.execution_status === 'executing' || plan.execution_status === 'completed'}
              className="flex items-center space-x-2"
            >
              <Play className="w-4 h-4" />
              {startMutation.isPending ? 'Starting...' :
               plan.execution_status === 'executing' ? 'Executing...' :
               plan.execution_status === 'completed' ? 'Completed' :
               !isPlanExecutable ? 'Robots Unavailable' :
               'Execute Plan'}
            </Button>
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
                  className="px-3 py-1.5 bg-blue-500/20 border border-blue-500/40 text-blue-300 rounded-lg text-sm font-semibold hover:bg-blue-500/30 transition-colors"
                  onClick={() => {
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
                  className="px-3 py-1.5 bg-purple-500/20 border border-purple-500/40 text-purple-300 rounded-lg text-sm font-semibold hover:bg-purple-500/30 transition-colors"
                  onClick={() => {
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
                    executionReady ? 'bg-green-500/20 border-green-500/40 text-green-300' :
                    allocationStatusValue === 'fully_allocated' ? 'bg-yellow-500/20 border-yellow-500/40 text-yellow-300' :
                    'bg-red-500/20 border-red-500/40 text-red-300'
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
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Plan Name
                </label>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  placeholder="Enter plan name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Description
                </label>
                <textarea
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                  rows={2}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 resize-none"
                  placeholder="Enter plan description"
                />
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <h2 className="text-xl font-bold text-white">
                {plan.name}
              </h2>
              <div className="text-slate-300 text-base leading-relaxed">
                {plan.description}
              </div>
            </div>
          )}

          {/* Bottom Row: Created date + Download Button */}
          <div className="flex items-center justify-between mt-4">
            {plan.created_at && (
              <div className="text-slate-500 text-xs">
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
          <div className="border-b border-slate-700">
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
                        ? 'bg-slate-800 text-white border-b-2 border-blue-500'
                        : 'text-slate-400 hover:text-slate-300'
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
                      <Card className="p-6 bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20">
                        <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                            <Bot className="w-4 h-4 text-emerald-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Robot Fleet Utilization</h3>
                        </div>

                        {/* Top metrics in horizontal layout */}
                        <div className="grid grid-cols-3 gap-4 mb-6">
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-emerald-300 mb-1">
                              {robotTypes.length}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Robot Types Used
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-emerald-300 mb-1">
                              {robotIds.length}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Robots Assigned
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-blue-300 mb-1">
                              {tasks.length > 0 ? Math.round((assignedTasks.length / tasks.length) * 100) : 0}%
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Assignment Rate
                            </div>
                          </div>
                        </div>

                        {/* Bottom details with styled sections */}
                        {(robotTypes.length > 0 || robotIds.length > 0) && (
                          <div className="space-y-4">
                            {robotTypes.length > 0 && (
                              <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-600/30">
                                <div className="flex items-center space-x-2 mb-3">
                                  <div className="w-5 h-5 bg-slate-500/20 rounded border border-slate-500/40 flex items-center justify-center">
                                    <span className="text-xs font-bold text-slate-300">T</span>
                                  </div>
                                  <h4 className="text-sm font-semibold text-slate-300">Robot Types Used</h4>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  {robotTypes.map(type => (
                                    <span key={type} className="px-3 py-2 bg-slate-500/20 border border-slate-500/40 text-slate-300 text-sm font-medium rounded-lg">
                                      {type}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                            {robotIds.length > 0 && (
                              <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-600/30">
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
                      <Card className="p-6 bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
                        <div>
                          <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                            <GitBranch className="w-4 h-4 text-blue-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Task Distribution</h3>
                        </div>

                        {/* Top metrics in horizontal layout */}
                        <div className="grid grid-cols-3 gap-4 mb-6">
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-blue-300 mb-1">
                              {tasks.length}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Total Tasks
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-emerald-300 mb-1">
                              {assignedTasks.length}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Assigned Tasks
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-amber-300 mb-1">
                              {unassignedTasks.length}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Unassigned Tasks
                            </div>
                          </div>
                        </div>
                          {(Object.keys(tasksByType).length > 1 || Object.keys(tasksByRobotId).length > 0) && (
                            <div className="pt-3 border-t border-slate-600/50 space-y-4">
                              {Object.keys(tasksByType).length > 1 && (
                                <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-600/30">
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
                                          <div key={type} className="flex items-center justify-between p-2 bg-slate-700/30 rounded border border-slate-600/20">
                                            <div className="flex items-center space-x-2">
                                              <span className="text-white text-sm font-medium px-2 py-1 bg-slate-600/50 rounded">
                                                {type}
                                              </span>
                                              <span className="text-slate-400 text-xs">
                                                {percentage.toFixed(0)}%
                                              </span>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                              <div className="w-12 bg-slate-600/30 rounded-full h-2">
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
                                <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-600/30">
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
                                          <div key={robotId} className="flex items-center justify-between p-2 bg-slate-700/30 rounded border border-slate-600/20">
                                            <div className="flex items-center space-x-2">
                                              <span className="text-white text-sm font-medium px-2 py-1 bg-emerald-600/50 rounded">
                                                {robotId}
                                              </span>
                                              <span className="text-slate-400 text-xs">
                                                {percentage.toFixed(0)}%
                                              </span>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                              <div className="w-12 bg-slate-600/30 rounded-full h-2">
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
                      <Card className="p-6 bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20">
                        <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                            <Wand2 className="w-4 h-4 text-purple-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Execution Readiness</h3>
                        </div>

                        {/* Top metrics in horizontal layout */}
                        <div className="grid grid-cols-3 gap-4">
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                               <div className={`text-2xl font-bold mb-1 ${
                                 allocationStatusValue === 'fully_allocated' ? 'text-blue-300' :
                                 allocationStatusValue === 'partially_allocated' ? 'text-yellow-300' :
                                 allocationStatusValue === 'unallocated' ? 'text-red-300' : 'text-slate-300'
                               }`}>
                              {capitalize(allocationStatusValue?.replace('_', ' ') || 'Unknown')}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Plan Status
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className={`text-2xl font-bold mb-1 ${
                              executionReady ? 'text-green-300' :
                              allocationStatusValue === 'fully_allocated' ? 'text-yellow-300' : 'text-red-300'
                            }`}>
                              {executionReady ? 'Yes' :
                               allocationStatusValue === 'fully_allocated' ? 'Robots Offline' : 'No'}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Execution Status
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-slate-300 mb-1">
                              {tasks.length > 0 ? Math.round((tasks.filter(t => !t.dependency_task_ids?.length).length / tasks.length) * 100) : 0}%
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
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
                      <Card className="p-6 bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20">
                        <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                            <Target className="w-4 h-4 text-emerald-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Goals Overview</h3>
                        </div>

                        <div className="space-y-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                              <div className="text-2xl font-bold text-emerald-300 mb-1">
                                {plan?.goal_ids?.length || 0}
                              </div>
                              <div className="text-slate-400 text-xs font-medium">
                                Total Goals
                              </div>
                            </div>
                            <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                              <div className="text-2xl font-bold text-emerald-300 mb-1">
                                {(() => {
                                  const goalTasks = tasks.filter(task => task.goal_id !== undefined)
                                  return goalTasks.length
                                })()}
                              </div>
                              <div className="text-slate-400 text-xs font-medium">
                                Tasks with Goals
                              </div>
                            </div>
                          </div>

                          {plan?.goal_ids && plan.goal_ids.length > 0 && (
                            <div>
                              <div className="text-sm font-medium text-slate-300 mb-3">Goal IDs:</div>
                              <div className="flex flex-wrap gap-2">
                                {plan.goal_ids.map(goalId => {
                                  const taskCount = tasks.filter((task: any) => task.goal_id === goalId).length
                                  return (
                                    <div key={goalId} className="bg-slate-700/50 px-3 py-2 rounded-lg border border-slate-600/50">
                                      <div className="text-sm font-medium text-emerald-300">#{goalId}</div>
                                      <div className="text-xs text-slate-400">{taskCount} tasks</div>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      </Card>

                      {/* Research Metrics */}
                      <Card className="p-6 bg-gradient-to-br from-orange-500/10 to-orange-600/5 border-orange-500/20">
                        <div className="flex items-center space-x-3 mb-6">
                          <div className="w-8 h-8 bg-orange-500/20 rounded-lg flex items-center justify-center">
                            <Link className="w-4 h-4 text-orange-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Research Metrics</h3>
                        </div>

                        {/* Top metrics in horizontal layout */}
                        <div className="grid grid-cols-3 gap-4">
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-orange-300 mb-1">
                              {Math.max(...tasks.map(t => t.dependency_task_ids?.length || 0), 0)}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Max Dependencies
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-orange-300 mb-1">
                              {tasks.length > 0 ? (tasks.reduce((sum, t) => sum + (t.dependency_task_ids?.length || 0), 0) / tasks.length).toFixed(1) : 0}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
                              Avg Dependencies
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50 text-center">
                            <div className="text-2xl font-bold text-orange-300 mb-1">
                              {tasks.filter(t => !t.dependency_task_ids?.length).length}
                            </div>
                            <div className="text-slate-400 text-xs font-medium">
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
                    <button
                      onClick={() => setTaskView('vertical')}
                      className={cn(
                        'px-3 py-1 text-sm rounded transition-colors',
                        taskView === 'vertical'
                          ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                          : 'text-slate-400 hover:text-slate-300'
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
                          : 'text-slate-400 hover:text-slate-300'
                      )}
                    >
                      DAG View
                    </button>
                    <button
                      onClick={downloadDAG}
                      className="px-3 py-1 text-sm rounded transition-colors text-slate-400 hover:text-slate-300 border border-slate-600 hover:border-slate-500"
                    >
                      📥 Download DAG (SVG)
                    </button>
                  </div>
                </div>

                {taskView === 'vertical' && <VerticalTaskList tasks={plan.tasks || []} />}
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
                        <Card key={goalId} className="p-4 bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20">
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
                              <p className="text-slate-300 mb-3">{goal?.description || 'Goal description not available'}</p>

                              {goalTasks.length > 0 && (
                                <div>
                                  <div className="text-sm font-medium text-slate-400 mb-2">Tasks in this plan:</div>
                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                    {goalTasks.map(task => (
                                      <div key={task.task_id} className="bg-slate-800/50 px-3 py-2 rounded border border-slate-600/50">
                                        <div className="flex items-center justify-between">
                                          <span className="text-sm font-medium text-slate-300">Task #{task.task_id}</span>
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
                                        <div className="text-xs text-slate-400 mt-1 truncate">{task.description}</div>
                                        {task.robot_id && (
                                          <div className="text-xs text-slate-500 mt-1">Robot: {task.robot_id}</div>
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
                          : 'text-slate-400 hover:text-slate-300'
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
                          : 'text-slate-400 hover:text-slate-300'
                      )}
                    >
                      Allocation
                    </button>
                  </div>
                </div>

                {promptsView === 'planning' && (
                  <Card className="p-6 bg-gradient-to-br from-blue-500/5 to-blue-600/5 border-blue-500/20">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                          <GitBranch className="w-4 h-4 text-blue-400" />
                        </div>
                        <h4 className="text-lg font-bold text-white">Planning Prompts</h4>
                      </div>
                      <button
                        onClick={downloadPlanningPrompts}
                        className="px-3 py-1 text-sm rounded transition-colors text-slate-400 hover:text-slate-300 border border-slate-600 hover:border-slate-500"
                      >
                        📥 Download (.zip)
                      </button>
                    </div>
                  {plan.planning_prompts ? (
                    <div className="space-y-4">
                      {typeof plan.planning_prompts === 'object' && plan.planning_prompts.system && plan.planning_prompts.user ? (
                        <div className="space-y-4">
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-6 h-6 bg-blue-500/20 rounded border border-blue-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-blue-300">S</span>
                              </div>
                              <h4 className="text-sm font-semibold text-blue-300">System Prompt</h4>
                            </div>
                            <div className="bg-slate-900/80 p-4 rounded border border-slate-700">
                              <pre className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                                {plan.planning_prompts.system}
                              </pre>
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-6 h-6 bg-green-500/20 rounded border border-green-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-green-300">U</span>
                              </div>
                              <h4 className="text-sm font-semibold text-green-300">User Prompt</h4>
                            </div>
                            <div className="bg-slate-900/80 p-4 rounded border border-slate-700">
                              <pre className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                                {plan.planning_prompts.user}
                              </pre>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50">
                          <div className="bg-slate-900/80 p-4 rounded border border-slate-700">
                            <pre className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                              {JSON.stringify(plan.planning_prompts, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="bg-slate-800/30 p-4 rounded-lg border border-slate-600/30">
                      <div className="flex items-center space-x-3 text-slate-400">
                        <div className="w-8 h-8 bg-slate-500/20 rounded-lg flex items-center justify-center">
                          <GitBranch className="w-4 h-4" />
                        </div>
                        <span>No planning prompts available for this plan.</span>
                      </div>
                    </div>
                  )}
                  </Card>
                )}

                {promptsView === 'allocation' && (
                  <Card className="p-6 bg-gradient-to-br from-purple-500/5 to-purple-600/5 border-purple-500/20">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                          <Wand2 className="w-4 h-4 text-purple-400" />
                        </div>
                        <h4 className="text-lg font-bold text-white">Allocation Prompts</h4>
                      </div>
                      <button
                        onClick={downloadAllocationPrompts}
                        className="px-3 py-1 text-sm rounded transition-colors text-slate-400 hover:text-slate-300 border border-slate-600 hover:border-slate-500"
                      >
                        📥 Download (.zip)
                      </button>
                    </div>
                  {plan.allocation_prompts ? (
                    <div className="space-y-4">
                      {typeof plan.allocation_prompts === 'object' && plan.allocation_prompts.system && plan.allocation_prompts.user ? (
                        <div className="space-y-4">
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-6 h-6 bg-purple-500/20 rounded border border-purple-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-purple-300">S</span>
                              </div>
                              <h4 className="text-sm font-semibold text-purple-300">System Prompt</h4>
                            </div>
                            <div className="bg-slate-900/80 p-4 rounded border border-slate-700">
                              <pre className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                                {plan.allocation_prompts.system}
                              </pre>
                            </div>
                          </div>
                          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50">
                            <div className="flex items-center space-x-2 mb-3">
                              <div className="w-6 h-6 bg-green-500/20 rounded border border-green-500/40 flex items-center justify-center">
                                <span className="text-xs font-bold text-green-300">U</span>
                              </div>
                              <h4 className="text-sm font-semibold text-green-300">User Prompt</h4>
                            </div>
                            <div className="bg-slate-900/80 p-4 rounded border border-slate-700">
                              <pre className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                                {plan.allocation_prompts.user}
                              </pre>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50">
                          <div className="bg-slate-900/80 p-4 rounded border border-slate-700">
                            <pre className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                              {JSON.stringify(plan.allocation_prompts, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="bg-slate-800/30 p-4 rounded-lg border border-slate-600/30">
                      <div className="flex items-center space-x-3 text-slate-400">
                        <div className="w-8 h-8 bg-slate-500/20 rounded-lg flex items-center justify-center">
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
                          : 'text-slate-400 hover:text-slate-300'
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
                          : 'text-slate-400 hover:text-slate-300'
                      )}
                    >
                      Allocation
                    </button>
                  </div>
                </div>

                {artifactsView === 'planning' && (
                  <Card className="p-6 bg-gradient-to-br from-blue-500/5 to-blue-600/5 border-blue-500/20">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                          <Link className="w-4 h-4 text-blue-400" />
                        </div>
                        <h4 className="text-lg font-bold text-white">Planning Artifacts</h4>
                      </div>
                      <button
                        onClick={downloadPlanningArtifacts}
                        className="px-3 py-1 text-sm rounded transition-colors text-slate-400 hover:text-slate-300 border border-slate-600 hover:border-slate-500"
                      >
                        📥 Download (.json)
                      </button>
                    </div>
                  {plan.planning_artifacts ? (
                    <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50">
                      <div className="bg-slate-900/80 p-4 rounded border border-slate-700">
                        <pre className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                          {JSON.stringify(plan.planning_artifacts, null, 2)}
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-800/30 p-4 rounded-lg border border-slate-600/30">
                      <div className="flex items-center space-x-3 text-slate-400">
                        <div className="w-8 h-8 bg-slate-500/20 rounded-lg flex items-center justify-center">
                          <Link className="w-4 h-4" />
                        </div>
                        <span>No planning artifacts available for this plan.</span>
                      </div>
                    </div>
                  )}
                  </Card>
                )}

                {artifactsView === 'allocation' && (
                  <Card className="p-6 bg-gradient-to-br from-purple-500/5 to-purple-600/5 border-purple-500/20">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                          <Link className="w-4 h-4 text-purple-400" />
                        </div>
                        <h4 className="text-lg font-bold text-white">Allocation Artifacts</h4>
                      </div>
                      <button
                        onClick={downloadAllocationArtifacts}
                        className="px-3 py-1 text-sm rounded transition-colors text-slate-400 hover:text-slate-300 border border-slate-600 hover:border-slate-500"
                      >
                        📥 Download (.json)
                      </button>
                    </div>
                  {plan.allocation_artifacts ? (
                    <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-600/50">
                      <div className="bg-slate-900/80 p-4 rounded border border-slate-700">
                        <pre className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                          {JSON.stringify(plan.allocation_artifacts, null, 2)}
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-800/30 p-4 rounded-lg border border-slate-600/30">
                      <div className="flex items-center space-x-3 text-slate-400">
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
    </>
  )
}