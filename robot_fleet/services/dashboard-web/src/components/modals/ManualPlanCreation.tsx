import React, { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Search, CheckCircle, Plus, Trash2, Bot, ArrowRight } from 'lucide-react'
import { Modal } from '../common/Modal'
import { Card } from '../common/Card'
import { Button } from '../common/Button'
import { goalsApi, robotsApi, plansApi } from '../../lib/api'
import { cn } from '../../lib/utils'

interface ManualPlanCreationProps {
  isOpen: boolean
  onClose: () => void
}

interface TaskData {
  temp_id: string
  description: string
  dependencies: string[]
  robot_id?: string
  goal_id?: number
}

export function ManualPlanCreation({ isOpen, onClose }: ManualPlanCreationProps) {
  const [selectedGoals, setSelectedGoals] = useState<number[]>([])
  const [goalSearch, setGoalSearch] = useState<string>('')
  const [planName, setPlanName] = useState<string>('')
  const [planDescription, setPlanDescription] = useState<string>('')
  const [goalAssignmentError, setGoalAssignmentError] = useState<string | null>(null)

  // Task creation state
  const [tasks, setTasks] = useState<TaskData[]>([])
  const [nextTaskId, setNextTaskId] = useState<number>(1)
  const [newTaskDescription, setNewTaskDescription] = useState<string>('')

  // DAG visualization state
  const [svgContent, setSvgContent] = useState<string>('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [zoom, setZoom] = useState(0.8)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 })
  const containerRef = React.useRef<HTMLDivElement>(null)

  // Load goals
  const { data: goals = [] } = useQuery({
    queryKey: ['goals'],
    queryFn: goalsApi.list,
  })

  // Load robots for allocation
  const { data: robots = [] } = useQuery({
    queryKey: ['robots'],
    queryFn: () => robotsApi.list(),
  })

  // Filter and search goals
  const filteredGoals = goals.filter(goal =>
    goalSearch === '' ||
    goal.goal_id.toString().includes(goalSearch) ||
    goal.description.toLowerCase().includes(goalSearch.toLowerCase())
  )

  // Task management functions
  const addTask = () => {
    if (!newTaskDescription.trim()) return

    const newTask: TaskData = {
      temp_id: `t${nextTaskId}`,
      description: newTaskDescription.trim(),
      dependencies: [],
      robot_id: undefined,
      goal_id: selectedGoals[0]
    }

    setTasks([...tasks, newTask])
    setNextTaskId(prev => prev + 1)
    setNewTaskDescription('')
  }

  const removeTask = (taskTempId: string) => {
    setTasks(tasks.filter(task => task.temp_id !== taskTempId))
  }

  const updateTask = (taskTempId: string, updates: Partial<TaskData>) => {
    setTasks(tasks.map(task =>
      task.temp_id === taskTempId ? { ...task, ...updates } : task
    ))
  }

  // Keyboard shortcuts for DAG
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'r' || e.key === 'R') {
        resetView()
      } else if (e.key === '+' || e.key === '=') {
        setZoom(prev => Math.min(3, prev * 1.2))
      } else if (e.key === '-') {
        setZoom(prev => Math.max(0.1, prev * 0.8))
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown)
    }

    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen])

  // Mouse event handlers for zoom and pan
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
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
    setZoom(0.8)
    setPan({ x: 0, y: 0 })
  }

  // Generate DAG when tasks change
  useEffect(() => {
    if (!tasks || tasks.length === 0) {
      setSvgContent('')
      setError(null)
      setIsGenerating(false)
      return
    }

    setIsGenerating(true)
    setError(null)

    // Generate Graphviz DOT format
    const generateDot = (tasks: TaskData[]) => {
      let dot = `digraph G {
  rankdir=LR;
  bgcolor="#ffffff";
  node [shape=plaintext, fontname="Arial"];
  edge [color="#334155",penwidth=3.0, arrowhead=vee, arrowsize=2.0, headclip=true, tailclip=true];
  graph [splines=spline, nodesep=2.0, ranksep=2.0];
`

      // Add nodes with custom HTML styling
      tasks.forEach(task => {
        // Simple status colors
        const statusColors = {
          border: '#64748b',
          bg: '#f8fafc',
          text: '#374151',
          headerBg: '#f3f4f6'
        }

        const taskId = task.temp_id.replace('t', '') // Show as "1", "2", "3" etc.
        const description = task.description.replace(/"/g, '\\"').replace(/</g, '&lt;').replace(/>/g, '&gt;')

        // Find robot data if assigned
        const assignedRobot = robots.find(r => r.robot_id === task.robot_id)
        const robotType = assignedRobot ? assignedRobot.robot_type : 'unassigned'
        const robotId = task.robot_id || ''

        // Calculate width for the visible content
        const badgeContentWidth = robotType.length * 16 + 24

        const descriptionWidth = Math.max(
          badgeContentWidth,
          300
        )

        const htmlLabel = `<TABLE BORDER="2" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" BGCOLOR="${statusColors.bg}" COLOR="${statusColors.border}" STYLE="ROUNDED">
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="${statusColors.headerBg}" CELLPADDING="12">
      <FONT COLOR="#1d4ed8" FACE="Arial" POINT-SIZE="40">
        <B>Task ${taskId}</B>
      </FONT>
    </TD>
  </TR>
  

  <TR>
    <TD ALIGN="CENTER" CELLPADDING="15" STYLE="max-width: ${descriptionWidth}px;">
      <FONT COLOR="${statusColors.text}" FACE="Arial" POINT-SIZE="36">
        <b>${description}</b>
      </FONT>
    </TD>
  </TR>

  <TR>
    <TD ALIGN="CENTER" CELLPADDING="12">
      <TABLE BORDER="0" CELLSPACING="6">
        <TR>
          <TD BGCOLOR="#7c3aed" CELLPADDING="10">
            <FONT COLOR="#ffffff" FACE="Arial" POINT-SIZE="32">
              <B>${robotType}</B>
            </FONT>
          </TD>
        </TR>
        ${task.robot_id ? `
        <TR>
          <TD BGCOLOR="#0d9488" CELLPADDING="10">
            <FONT COLOR="#ffffff" FACE="Arial" POINT-SIZE="32">
              <B>${robotId}</B>
            </FONT>
          </TD>
        </TR>
        ` : ''}
      </TABLE>
    </TD>
  </TR>
</TABLE>`

        dot += `  ${task.temp_id} [label=<${htmlLabel}>];\n`
      })

      // Add edges for dependencies
      tasks.forEach(task => {
        if (task.dependencies && task.dependencies.length > 0) {
          task.dependencies.forEach((depId: string) => {
            dot += `  ${depId} -> ${task.temp_id} [color="#374151", penwidth="2"];\n`
          })
        }
      })

      dot += '}'
      return dot
    }

    const dotSource = generateDot(tasks)

    // Generate SVG using Graphviz
    try {
      import('@hpcc-js/wasm-graphviz').then(async ({ Graphviz }) => {
        try {
          const graphviz = await Graphviz.load()
          const svg = graphviz.dot(dotSource, 'svg')

          if (svg && svg.length > 100 && svg.includes('<svg')) {
            setSvgContent(svg)
          } else {
            setError('Generated SVG is invalid')
          }
          setIsGenerating(false)
        } catch (loadError: any) {
          setError(`Graphviz failed: ${loadError.message}`)
          setIsGenerating(false)
        }
      }).catch((importError: any) => {
        setError(`Graphviz import failed: ${importError.message}`)
        setIsGenerating(false)
      })
    } catch (err: any) {
      setError(`Graphviz setup error: ${err.message}`)
      setIsGenerating(false)
    }
  }, [tasks])

  // Plan creation mutation
  const createPlanMutation = useMutation({
    mutationFn: plansApi.createManual,
    onSuccess: (result) => {
      console.log('Plan created successfully:', result)
      onClose()
      // TODO: Navigate to plan details or refresh plan list
    },
    onError: (error) => {
      console.error('Failed to create manual plan:', error)
      // TODO: Show error message to user
    }
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (selectedGoals.length === 0 || !planName.trim() || !planDescription.trim() || tasks.length === 0) return

    // Validate each selected goal is used by at least one task
    const unusedSelectedGoals = selectedGoals.filter(
      (goalId) => !tasks.some((t) => t.goal_id === goalId)
    )
    if (unusedSelectedGoals.length > 0) {
      setGoalAssignmentError(
        `You selected Goal${unusedSelectedGoals.length === 1 ? '' : 's'} ${unusedSelectedGoals
          .map((g) => `#${g}`)
          .join(', ')}, but ${unusedSelectedGoals.length === 1 ? 'it is' : 'they are'} not assigned to any task. ` +
          `Assign each selected goal to at least one task, or deselect it.`
      )
      return
    }

    // Format tasks for API
    const formattedTasks = tasks.map((task) => ({
      temp_id: task.temp_id,
      description: task.description,
      depends_on: task.dependencies,
      robot_id: task.robot_id || undefined,
      goal_id: task.goal_id || undefined
    }))

    if (tasks.some(t => !t.goal_id)) {
      setGoalAssignmentError('All tasks must be assigned to a goal before creating the plan.')
      return
    }

    const planData = {
      name: planName.trim(),
      description: planDescription.trim(),
      // goal_ids: selectedGoals,
      tasks: formattedTasks
    }

    console.log('Creating manual plan:', planData)
    createPlanMutation.mutate(planData)
  }

  const handleGoalToggle = (goalId: number) => {
    if (selectedGoals.includes(goalId)) {
      setSelectedGoals(selectedGoals.filter(id => id !== goalId))
    } else {
      setSelectedGoals([...selectedGoals, goalId])
    }
  }

  return (
    <>
      <Modal isOpen={isOpen} onClose={onClose} title="Create Plan Manually" size="wide">
        <form onSubmit={handleSubmit} className="space-y-6 max-h-[75vh] overflow-y-auto">
        {/* Plan Name and Description */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text)] mb-2">
              Plan Name <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={planName}
              onChange={(e) => setPlanName(e.target.value)}
              placeholder="Enter a descriptive name for your plan"
              className="w-full px-4 py-3 bg-surface-elevated/50 border border-border rounded-lg text-white placeholder-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-cyber-500 focus:border-transparent transition-all"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[var(--color-text)] mb-2">
              Description <span className="text-red-400">*</span>
            </label>
            <textarea
              value={planDescription}
              onChange={(e) => setPlanDescription(e.target.value)}
              rows={3}
              placeholder="Describe what this plan accomplishes..."
              className="w-full px-4 py-3 bg-surface-elevated/50 border border-border rounded-lg text-white placeholder-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-cyber-500 focus:border-transparent transition-all resize-none"
            />
          </div>
        </div>

        {/* Goals Selection - Beautiful Cards with Search */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white flex items-center gap-3">
              <span className="w-3 h-3 bg-gradient-to-r from-violet-400 to-pink-400 rounded-full"></span>
              Select Goals to Plan For
            </h3>
            <div className="text-sm text-[var(--color-text-secondary)]">
              {selectedGoals.length} of {filteredGoals.length} selected
            </div>
          </div>

          {/* Search Bar */}
          <div className="relative mb-6">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-[var(--color-text-secondary)]" />
            <input
              type="text"
              placeholder="Search goals by ID or description..."
              value={goalSearch}
              onChange={(e) => setGoalSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-surface-overlay/50 border border-border rounded-lg text-white placeholder-[var(--color-text-muted)] focus:border-cyber-500 focus:ring-1 focus:ring-cyber-500 transition-all"
            />
          </div>

          {/* Goals Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
            {filteredGoals.map((goal) => (
              <Card
                key={goal.goal_id}
                className={cn(
                  'cursor-pointer transition-all duration-200 p-4 group relative',
                  selectedGoals.includes(goal.goal_id)
                    ? 'ring-2 ring-cyber-500 border-cyber-500 bg-gradient-to-br from-cyber-500/10 to-emerald-500/10 shadow-lg shadow-cyber-500/20'
                    : 'hover:border-slate-500 hover:shadow-md hover:shadow-slate-500/10 border-border'
                )}
                onClick={() => handleGoalToggle(goal.goal_id)}
              >
                <div className="flex items-start gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-semibold text-white text-sm">
                        Goal #{goal.goal_id}
                      </h4>
                    </div>
                    <p className="text-sm text-[var(--color-text)] leading-relaxed line-clamp-3">
                      {goal.description}
                    </p>
                  </div>

                  {/* Selection Indicator */}
                  {selectedGoals.includes(goal.goal_id) && (
                    <div className="flex-shrink-0 w-6 h-6 bg-gradient-to-r from-cyber-500 to-emerald-500 rounded-full flex items-center justify-center shadow-lg">
                      <CheckCircle className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
              </Card>
            ))}
          </div>

          {filteredGoals.length === 0 && (
            <div className="text-center py-8 text-[var(--color-text-secondary)]">
              <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No goals match your search.</p>
            </div>
          )}
        </div>

        {/* Create and Allocate Tasks */}
        <div className="border-t border-border pt-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white flex items-center gap-3">
              <span className="w-3 h-3 bg-gradient-to-r from-amber-400 to-orange-400 rounded-full"></span>
              Create and Allocate Tasks
            </h3>
            <div className="text-sm text-[var(--color-text-secondary)]">
              {tasks.length} task{tasks.length !== 1 ? 's' : ''} created
            </div>
          </div>

          {/* Add New Task */}
          <div className="mb-6">
            <div className="flex gap-3">
              <div className="flex-1">
                <input
                  type="text"
                  value={newTaskDescription}
                  onChange={(e) => setNewTaskDescription(e.target.value)}
                  placeholder="Enter task description..."
                  className="w-full px-4 py-3 bg-surface-elevated/50 border border-border rounded-lg text-white placeholder-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-all"
                  onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTask())}
                />
              </div>
              <Button
                type="button"
                onClick={addTask}
                disabled={!newTaskDescription.trim()}
                className="px-4 py-3 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 disabled:opacity-50"
              >
                <Plus className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Task List */}
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {tasks.map((task, index) => (
              <Card key={task.temp_id} className="p-4 border-border">
                <div className="flex items-start gap-4">
                  {/* Task Number */}
                  <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-r from-amber-500 to-orange-500 rounded-full flex items-center justify-center text-white font-semibold text-sm">
                    {index + 1}
                  </div>

                  <div className="flex-1 space-y-3">
                    {/* Task Description */}
                    <div>
                      <textarea
                        value={task.description}
                        onChange={(e) => updateTask(task.temp_id, { description: e.target.value })}
                        rows={2}
                        className="w-full px-3 py-2 bg-surface-overlay/50 border border-border rounded text-white placeholder-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-all resize-none text-sm"
                        placeholder="Task description..."
                      />
                    </div>

                    {/* Dependencies and Robot Allocation */}
                    <div className="flex items-center gap-4 flex-wrap">
                      {/* Dependencies */}
                      <div className="flex items-center gap-2">
                        <ArrowRight className="w-4 h-4 text-[var(--color-text-secondary)]" />
                        <select
                          multiple
                          value={task.dependencies}
                          onChange={(e) => {
                            const selectedOptions = Array.from(e.target.selectedOptions, option => option.value)
                            updateTask(task.temp_id, { dependencies: selectedOptions })
                          }}
                          className="px-3 py-1 bg-surface-overlay/50 border border-border rounded text-white text-xs focus:outline-none focus:ring-1 focus:ring-amber-500"
                        >
                          {tasks.slice(0, index).map((prevTask) => (
                            <option key={prevTask.temp_id} value={prevTask.temp_id}>
                              Task {prevTask.temp_id.replace('t', '')}
                            </option>
                          ))}
                        </select>
                        {task.dependencies.length > 0 && (
                          <span className="text-xs text-amber-400">
                            Depends on {task.dependencies.length} task{task.dependencies.length !== 1 ? 's' : ''}
                          </span>
                        )}
                      </div>

                      {/* Robot Allocation */}
                      <div className="flex items-center gap-2">
                        <Bot className="w-4 h-4 text-[var(--color-text-secondary)]" />
                        <select
                          value={task.robot_id || ''}
                          onChange={(e) => updateTask(task.temp_id, { robot_id: e.target.value || undefined })}
                          className="px-3 py-1 bg-surface-overlay/50 border border-border rounded text-white text-xs focus:outline-none focus:ring-1 focus:ring-amber-500"
                        >
                          <option value="">No robot assigned</option>
                          {robots.map((robot) => (
                            <option key={robot.robot_id} value={robot.robot_id}>
                              {robot.robot_id} ({robot.robot_type})
                            </option>
                          ))}
                        </select>
                        {task.robot_id && (
                          <span className="text-xs text-emerald-400 flex items-center gap-1">
                            <Bot className="w-3 h-3" />
                            Assigned
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Goal Allocation */}
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-[var(--color-text-secondary)]" />
                    <select
                      value={task.goal_id ?? ''}
                      onChange={(e) =>
                        updateTask(task.temp_id, {
                          goal_id: e.target.value ? Number(e.target.value) : undefined
                        })
                      }
                      className="px-3 py-1 bg-surface-overlay/50 border border-border rounded text-white text-xs focus:outline-none focus:ring-1 focus:ring-cyber-500"
                    >
                      <option value="">No goal assigned</option>
                      {selectedGoals.map((goalId) => {
                        const goal = goals.find(g => g.goal_id === goalId)
                        if (!goal) return null

                        return (
                          <option key={goal.goal_id} value={goal.goal_id}>
                            Goal #{goal.goal_id}
                          </option>
                        )
                      })}
                    </select>

                    {task.goal_id && (
                      <span className="text-xs text-cyber-400">
                        Goal #{task.goal_id}
                      </span>
                    )}
                  </div>

                  {/* Delete Button */}
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => removeTask(task.temp_id)}
                    className="text-red-400 hover:text-red-300 hover:bg-red-500/10 p-2"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </Card>
            ))}
          </div>

          {tasks.length === 0 && (
            <div className="text-center py-8 text-[var(--color-text-secondary)]">
              <Plus className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No tasks created yet. Add your first task above.</p>
            </div>
          )}
        </div>

        {/* Task Dependency Graph */}
        {tasks.length > 0 && (
          <div className="border-t border-border pt-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white flex items-center gap-3">
                <span className="w-3 h-3 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full"></span>
                Task Dependency Graph
              </h3>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-[var(--color-text-muted)]">
                  Zoom: {Math.round(zoom * 100)}%
                </span>
                <Button
                  type="button"
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
              style={{ height: '400px', cursor: isDragging ? 'grabbing' : 'grab' }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onWheel={handleWheel}
            >
              {isGenerating ? (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-slate-400 mx-auto mb-2"></div>
                    <p className="text-[var(--color-text-muted)]">Generating DAG visualization...</p>
                  </div>
                </div>
              ) : error ? (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center text-red-600">
                    <p>Error: {error}</p>
                  </div>
                </div>
              ) : svgContent ? (
                <div
                  className="w-full h-full flex items-center justify-center p-4"
                  style={{
                    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    transformOrigin: 'center center',
                    transition: isDragging ? 'none' : 'transform 0.1s ease-out'
                  }}
                  dangerouslySetInnerHTML={{ __html: svgContent }}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center text-[var(--color-text-secondary)]">
                    <p>Add tasks above to see the dependency graph</p>
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center justify-center gap-2 flex-wrap mt-4">
              <span className="text-[10px] text-[var(--color-text-muted)]">Arrows = dependencies</span>
              <span className="text-[10px] text-[var(--color-text-muted)]">Drag to pan</span>
              <span className="text-[10px] text-[var(--color-text-muted)]">Scroll to zoom</span>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end gap-3 pt-4 border-t border-border">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="submit"
            disabled={selectedGoals.length === 0 || !planName.trim() || !planDescription.trim() || tasks.length === 0 || createPlanMutation.isPending}
            className="bg-gradient-to-r from-cyber-500 to-emerald-500 hover:from-cyber-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {createPlanMutation.isPending ? 'Creating Plan...' : 'Create Manual Plan'}
          </Button>
        </div>
        </form>
      </Modal>

      {/* Validation Error Modal */}
      <Modal
        isOpen={goalAssignmentError !== null}
        onClose={() => setGoalAssignmentError(null)}
        title="Goal assignment required"
        size="md"
      >
        <div className="space-y-4">
          <p className="text-sm text-[var(--color-text)]">{goalAssignmentError}</p>
          <div className="flex justify-end">
            <Button variant="primary" onClick={() => setGoalAssignmentError(null)}>
              OK
            </Button>
          </div>
        </div>
      </Modal>
    </>
  )
}