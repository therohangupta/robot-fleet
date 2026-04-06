import { useState, useEffect, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Target, Plus, Trash2, Search, ChevronDown, Info, GitBranch, Bot, CheckCircle, Play, AlertTriangle, XCircle } from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { goalsApi, plansApi, methodsApi, tasksApi, robotsApi, useRealtimeUpdates } from '../lib/api'
import { cn, getPlanningStrategyName, getAllocationStrategyName, setMethodData } from '../lib/utils'

interface RobotHealth {
  robot_id: string
  host: string
  port: number
  reachable: boolean
  latency_ms?: number
  error?: string
}

// =============================================================================
// Goal Details Modal with Tabs
// =============================================================================

function GoalDetailsModal({ goalId, isOpen, onClose }: { goalId: number | null; isOpen: boolean; onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<'overview' | 'plans' | 'robots'>('overview')

  const { data: goals = [] } = useQuery({
    queryKey: ['goals'],
    queryFn: goalsApi.list,
  })

  const { data: plans = [] } = useQuery({
    queryKey: ['plans'],
    queryFn: plansApi.list,
  })

  const { data: robots = [] } = useQuery({
    queryKey: ['robots'],
    queryFn: () => robotsApi.list(),
  })

  // Enable real-time updates for robot health
  useRealtimeUpdates()

  const { data: robotHealth = {} } = useQuery({
    queryKey: ['robot-health'],
    queryFn: async () => {
      try {
        const response = await fetch('/api/robots/health/all')
        const data = await response.json()
        const healthMap: Record<string, RobotHealth> = {}
        for (const health of data.robots || []) {
          healthMap[health.robot_id] = health
        }
        return healthMap
      } catch (error) {
        console.error('Failed to fetch robot health:', error)
        return {}
      }
    },
    // Enable background refetching for more responsive updates
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
    staleTime: 1000, // Consider data stale after 1 second
    // WebSocket handles most updates, this is backup
  })

  const goal = goals.find(g => g.goal_id === goalId)
  const goalPlans = goalId ? plans.filter(plan => plan.goal_ids.includes(goalId)) : []

  // Get unique robots assigned to this goal across all plans
  const assignedRobots = new Set<string>()
  goalPlans.forEach(plan => {
    if (plan.allocation_artifacts?.final_allocation?.allocations) {
      plan.allocation_artifacts.final_allocation.allocations.forEach((alloc: any) => {
        assignedRobots.add(alloc.robot_id)
      })
    }
  })
  const goalRobots = robots.filter(robot => assignedRobots.has(robot.robot_id))

  const tabs = [
    { id: 'overview', label: 'Overview', icon: <Info className="w-4 h-4" /> },
    { id: 'plans', label: 'Plans', icon: <GitBranch className="w-4 h-4" /> },
    { id: 'robots', label: 'Robots', icon: <Bot className="w-4 h-4" /> },
  ]

  if (!goal) return null

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Goal #${goal.goal_id}`} size="wide" className="max-h-[90vh]">
      <div className="space-y-6 max-h-[75vh] overflow-y-auto">
        {/* Tab Navigation */}
        <div className="flex space-x-1 border-b border-border pb-2 overflow-x-auto">
          {tabs.map((tab) => (
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
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Goal Description */}
            <Card className="p-6">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Target className="w-6 h-6 text-purple-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2">Goal Description</h3>
                  <p className="text-[var(--color-text)] leading-relaxed">{goal.description}</p>
                </div>
              </div>
            </Card>

            {/* Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-4 border-border">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <GitBranch className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-blue-400">{goalPlans.length}</div>
                    <div className="text-sm text-[var(--color-text-secondary)]">Plans Generated</div>
                  </div>
                </div>
              </Card>

              <Card className="p-4 border-border">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-emerald-400">
                      {goalPlans.reduce((total, plan) => {
                        if (plan.allocation_artifacts?.task_descriptions) {
                          return total + plan.allocation_artifacts.task_descriptions.filter((task: any) => task.goal_id === goalId?.toString()).length
                        }
                        return total
                      }, 0)}
                    </div>
                    <div className="text-sm text-[var(--color-text-secondary)]">Tasks Created</div>
                  </div>
                </div>
              </Card>

              <Card className="p-4 border-border">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <Bot className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-purple-400">{goalRobots.length}</div>
                    <div className="text-sm text-[var(--color-text-secondary)]">Robots Assigned</div>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'plans' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Generated Plans</h3>
              <span className="text-sm text-[var(--color-text-muted)] bg-surface-overlay px-3 py-1 rounded-full">
                {goalPlans.length} plans
              </span>
            </div>

            {goalPlans.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2">
                {goalPlans.map(plan => {
                  const executionStatus = plan.execution_status || 'not_executed'
                  const goalCount = Array.isArray(plan.goal_ids) ? new Set(plan.goal_ids).size : 0

                  let badgeConfig = { bg: 'bg-surface-elevated/30', border: 'border-border', text: 'text-[var(--color-text-secondary)]', label: 'Unknown', icon: AlertTriangle }

                  if (executionStatus === 'completed') {
                    badgeConfig = { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', label: 'Completed', icon: CheckCircle }
                  } else if (executionStatus === 'executing') {
                    badgeConfig = { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', label: 'Running', icon: Play }
                  } else {
                    badgeConfig = { bg: 'bg-surface-elevated/30', border: 'border-border', text: 'text-[var(--color-text-secondary)]', label: 'Not Started', icon: AlertTriangle }
                  }

                  return (
                    <Card key={plan.plan_id} className="p-4 hover:bg-surface-overlay/50 transition-colors cursor-pointer"
                          onClick={() => window.open(`/plans/${plan.plan_id}`, '_blank')}>
                      {/* Plan Header */}
                      <div className="flex items-center justify-between mb-3">
                        <span className="px-2 py-1 bg-yellow-500/20 border border-yellow-500/40 text-yellow-300 rounded text-xs font-semibold">
                          P{plan.plan_id}
                        </span>
                        <div className={cn('px-2 py-1 rounded text-xs flex items-center gap-1 font-medium', badgeConfig.bg, badgeConfig.text)}>
                          <badgeConfig.icon className="w-3 h-3" />
                          {badgeConfig.label}
                        </div>
                      </div>

                      {/* Plan Name & Description */}
                      <div className="mb-3">
                        <h4 className="font-medium text-white mb-1">{plan.name || `Plan ${plan.plan_id}`}</h4>
                        {plan.description && (
                          <p className="text-sm text-[var(--color-text-secondary)] line-clamp-2">{plan.description}</p>
                        )}
                      </div>

                      {/* Strategy Tags */}
                      <div className="flex items-center gap-2 mb-3">
                        <button className="px-2 py-1 bg-blue-500/20 border border-blue-500/40 text-blue-300 rounded text-xs font-semibold hover:bg-blue-500/30 transition-colors">
                          {getPlanningStrategyName(plan.planning_strategy)}
                        </button>
                        {plan.allocation_strategy && plan.allocation_strategy !== 4 && (
                          <button className="px-2 py-1 bg-purple-500/20 border border-purple-500/40 text-purple-300 rounded text-xs font-semibold hover:bg-purple-500/30 transition-colors">
                            {getAllocationStrategyName(plan.allocation_strategy)}
                          </button>
                        )}
                      </div>

                      {/* Statistics */}
                      <div className="grid grid-cols-3 gap-2">
                        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-2 text-center">
                          <div className="text-lg font-bold text-emerald-300 mb-1">
                            {goalCount}
                          </div>
                          <div className="text-xs text-[var(--color-text-secondary)]">Goals</div>
                        </div>
                        <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2 text-center">
                          <div className="text-lg font-bold text-blue-300 mb-1">
                            {Array.isArray(plan.task_ids) ? plan.task_ids.length : 0}
                          </div>
                          <div className="text-xs text-[var(--color-text-secondary)]">Tasks</div>
                        </div>
                        <div className="bg-purple-500/10 border border-purple-500/20 rounded p-2 text-center">
                          <div className="text-lg font-bold text-purple-300 mb-1">
                            {plan.allocation_artifacts?.final_allocation?.allocations?.length || 0}
                          </div>
                          <div className="text-xs text-[var(--color-text-secondary)]">Robots</div>
                        </div>
                      </div>
                    </Card>
                  )
                })}
              </div>
            ) : (
              <EmptyState
                icon={<GitBranch className="w-12 h-12 text-[var(--color-text-muted)]" />}
                title="No Plans Generated"
                description="No plans have been created to achieve this goal yet."
              />
            )}
          </div>
        )}

        {activeTab === 'robots' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Assigned Robots</h3>
              <span className="text-sm text-[var(--color-text-muted)] bg-surface-overlay px-3 py-1 rounded-full">
                {goalRobots.length} robots
              </span>
            </div>

            {goalRobots.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {goalRobots.map(robot => {
                  const health = robotHealth[robot.robot_id]
                  const isReachable = health?.reachable === true

                  return (
                    <Card key={robot.robot_id} hover className="relative overflow-hidden cursor-pointer" onClick={() => window.open(`/robots?robot=${robot.robot_id}`, '_blank')}>
                      <div className={cn(
                        'absolute top-0 left-0 w-1 h-full',
                        isReachable ? 'bg-emerald-500' : 'bg-red-500'
                      )} />

                      <div className="pl-4">
                        {/* Header */}
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div className={cn(
                              'w-8 h-8 rounded-lg flex items-center justify-center',
                              isReachable ? 'bg-emerald-500/10' : 'bg-surface-overlay'
                            )}>
                              <Bot className={cn('w-4 h-4', isReachable ? 'text-emerald-400' : 'text-[var(--color-text-muted)]')} />
                            </div>
                            <div>
                              <h3 className="font-medium text-white">{robot.robot_id}</h3>
                              <p className="text-xs text-[var(--color-text-muted)]">{robot.robot_type}</p>
                            </div>
                          </div>

                          {/* Connection Status */}
                          <div className="flex items-center gap-1.5">
                            {isReachable ? (
                              <CheckCircle className="w-4 h-4 text-emerald-400" />
                            ) : (
                              <XCircle className="w-4 h-4 text-red-400" />
                            )}
                            <span className={cn(
                              'text-xs font-medium',
                              isReachable ? 'text-emerald-400' : 'text-red-400'
                            )}>
                              {isReachable ? 'Connected' : 'Unreachable'}
                            </span>
                          </div>
                        </div>

                        {/* Capabilities */}
                        <div className="flex flex-wrap gap-1 mt-2">
                          {robot.capabilities.slice(0, 3).map((capability: string, index: number) => (
                            <span
                              key={index}
                              className="px-2 py-0.5 bg-surface-elevated/50 text-[var(--color-text)] rounded text-xs font-medium"
                            >
                              {capability}
                            </span>
                          ))}
                          {robot.capabilities.length > 3 && (
                            <span className="px-2 py-0.5 bg-surface-elevated/50 text-[var(--color-text-secondary)] rounded text-xs">
                              +{robot.capabilities.length - 3} more
                            </span>
                          )}
                        </div>

                        {/* Connection Info */}
                        <div className="flex items-center justify-between pt-3 mt-3 border-t border-border">
                          <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
                            <span className="font-mono">
                              {robot.task_server_info?.host}:{robot.task_server_info?.port}
                            </span>
                            {health?.latency_ms && (
                              <span className="text-emerald-400">({Math.round(health.latency_ms)}ms)</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </Card>
                  )
                })}
              </div>
            ) : (
              <EmptyState
                icon={<Bot className="w-12 h-12 text-[var(--color-text-muted)]" />}
                title="No Robots Assigned"
                description="No robots have been assigned to solve this goal yet."
              />
            )}
          </div>
        )}
      </div>
    </Modal>
  )
}

function CreateGoalModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [description, setDescription] = useState('')
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: goalsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['goals'] })
      onClose()
      setDescription('')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({ description })
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Create New Goal">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-[var(--color-text)] mb-2">
            Goal Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe what you want the robots to accomplish..."
            rows={4}
            className="w-full px-4 py-3 bg-surface-overlay border border-border rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 resize-none"
            required
          />
        </div>

        {mutation.error && (
          <p className="text-sm text-red-400">{(mutation.error as Error).message}</p>
        )}

        <div className="flex justify-end gap-3 pt-4">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? 'Creating...' : 'Create Goal'}
          </Button>
        </div>
      </form>
    </Modal>
  )
}

export function Goals() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [plansModalGoal, setPlansModalGoal] = useState<number | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState<'goal_id' | 'created_date'>('goal_id')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const queryClient = useQueryClient()

  const { data: goals = [], isLoading } = useQuery({
    queryKey: ['goals'],
    queryFn: goalsApi.list,
  })

  const { data: plans = [] } = useQuery({
    queryKey: ['plans'],
    queryFn: plansApi.list,
  })

  const { data: tasks = [] } = useQuery({
    queryKey: ['tasks'],
    queryFn: () => tasksApi.list(),
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

  // Calculate plans count for each goal
  const getPlansCountForGoal = (goalId: number) => {
    return plans.filter(plan => plan.goal_ids.includes(goalId)).length
  }

  // Calculate unique robots count for plans assigned to a goal
  const getRobotsCountForGoal = (goalId: number) => {
    const goalPlans = plans.filter(plan => plan.goal_ids.includes(goalId))
    const uniqueRobots = new Set<string>()

    goalPlans.forEach(plan => {
      const planTasks = tasks.filter(task => task.plan_id === plan.plan_id && task.robot_id)
      planTasks.forEach(task => {
        if (task.robot_id) {
          uniqueRobots.add(task.robot_id)
        }
      })
    })

    return uniqueRobots.size
  }

  const deleteMutation = useMutation({
    mutationFn: goalsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['goals'] })
    },
  })

  // Filter and sort goals
  const filteredAndSortedGoals = useMemo(() => {
    let filtered = goals

    // Apply search filter
    if (searchTerm.trim()) {
      filtered = filtered.filter(goal =>
        goal.description.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    // Apply sorting
    const sorted = [...filtered].sort((a, b) => {
      let aValue: number
      let bValue: number

      if (sortBy === 'goal_id') {
        aValue = a.goal_id
        bValue = b.goal_id
      } else {
        // Since we don't have created_at, use goal_id as proxy for creation order
        aValue = a.goal_id
        bValue = b.goal_id
      }

      if (sortOrder === 'asc') {
        return aValue - bValue
      } else {
        return bValue - aValue
      }
    })

    return sorted
  }, [goals, searchTerm, sortBy, sortOrder])

  if (isLoading) {
    return <div className="text-[var(--color-text-secondary)]">Loading...</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Goals</h1>
          <p className="text-[var(--color-text-secondary)]">
            {filteredAndSortedGoals.length} of {goals.length} goals
            {searchTerm && ` matching "${searchTerm}"`}
          </p>
        </div>
        <Button onClick={() => setIsModalOpen(true)}>
          <Plus className="w-4 h-4" />
          Create Goal
        </Button>
      </div>

      {/* Search and Sort Controls */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        {/* Search Input */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-[var(--color-text-secondary)]" />
          <input
            type="text"
            placeholder="Search goals by description..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-surface-overlay/50 border border-border rounded-lg text-white placeholder-[var(--color-text-muted)] focus:outline-none focus:border-cyber-500 focus:ring-1 focus:ring-cyber-500/50"
          />
        </div>

        {/* Sort Controls */}
        <div className="flex gap-2">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'goal_id' | 'created_date')}
            className="px-3 py-2 bg-surface-overlay/50 border border-border rounded-lg text-white focus:outline-none focus:border-cyber-500 focus:ring-1 focus:ring-cyber-500/50"
          >
            <option value="goal_id">ID</option>
            <option value="created_date">Date Created</option>
          </select>

          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="px-3 py-2 bg-surface-overlay/50 border border-border rounded-lg text-white hover:bg-surface-elevated/50 transition-colors flex items-center gap-1"
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
            <ChevronDown className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Goals List */}
      {filteredAndSortedGoals.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredAndSortedGoals.map((goal) => (
            <Card key={goal.goal_id} className="p-4 hover:bg-surface-overlay/50 transition-colors cursor-pointer"
                  onClick={() => setPlansModalGoal(goal.goal_id)}>
              {/* Goal Header */}
              <div className="flex items-start justify-between mb-4">
                <span className="px-2 py-1 bg-purple-500/20 border border-purple-500/40 text-purple-300 rounded text-xs font-semibold">
                  G{goal.goal_id}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteMutation.mutate(goal.goal_id)
                  }}
                  className="text-red-400 hover:text-red-300 hover:bg-red-500/10 p-2 h-8 w-8"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>

              {/* Goal Description */}
              <div className="mb-4">
                <p className="text-sm text-[var(--color-text)] line-clamp-3">
                  {goal.description}
                </p>
              </div>

              {/* Statistics Boxes */}
              <div className="grid grid-cols-3 gap-2">
                {/* Tasks Box */}
                <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2 text-center">
                  <div className="text-lg font-bold text-blue-300 mb-1">
                    {goal.task_ids.length}
                  </div>
                  <div className="text-xs text-blue-400 font-medium">
                    Tasks
                  </div>
                </div>

                {/* Plans Box */}
                <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-2 text-center">
                  <div className="text-lg font-bold text-emerald-300 mb-1">
                    {getPlansCountForGoal(goal.goal_id)}
                  </div>
                  <div className="text-xs text-emerald-400 font-medium">
                    Plans
                  </div>
                </div>

                {/* Robots Box */}
                <div className="bg-yellow-500/10 border border-yellow-500/20 rounded p-2 text-center">
                  <div className="text-lg font-bold text-yellow-300 mb-1">
                    {getRobotsCountForGoal(goal.goal_id)}
                  </div>
                  <div className="text-xs text-yellow-400 font-medium">
                    Robots
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : goals.length === 0 ? (
        <EmptyState
          icon={<Target className="w-8 h-8" />}
          title="No goals created"
          description="Create your first goal to define what you want your robots to accomplish."
          action={
            <Button onClick={() => setIsModalOpen(true)}>
              <Plus className="w-4 h-4" />
              Create Goal
            </Button>
          }
        />
      ) : (
        <div className="text-center py-12 text-[var(--color-text-muted)]">
          <Search className="w-12 h-12 text-[var(--color-text-muted)] mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium text-[var(--color-text-secondary)] mb-2">No goals match your search</p>
          <p className="text-[var(--color-text-muted)] mb-4">Try adjusting your search terms or clearing the filter</p>
          <Button
            variant="secondary"
            onClick={() => setSearchTerm('')}
            className="mr-2"
          >
            Clear Search
          </Button>
        </div>
      )}

      <CreateGoalModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
      <GoalDetailsModal
        goalId={plansModalGoal}
        isOpen={plansModalGoal !== null}
        onClose={() => setPlansModalGoal(null)}
      />
    </div>
  )
}
