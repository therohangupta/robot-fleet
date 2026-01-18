import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Plus, Play, Trash2, CheckCircle,
  AlertCircle, Users, Zap, AlertTriangle, Copy, Search
} from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { plansApi, goalsApi, robotsApi, methodsApi } from '../lib/api'
import { cn } from '../lib/utils'
import { useRealtimeUpdates } from '../lib/api'
import type { Robot, Goal, PlanStatus } from '../types'
import type { MethodSummary } from '../lib/api'

interface LocalStrategy {
  id: string
  name: string
  type: 'planning' | 'allocation'
}


// Method Selection Card Component (compact, description-focused)
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
        'cursor-pointer hover:border-slate-600 transition-all p-3',
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
              method.method_type === 'manual' && 'bg-slate-500/10 border-slate-500/30 text-slate-400'
            )}>
              {method.method_type}
            </span>
            {isSelected && (
              <CheckCircle className="w-4 h-4 text-cyber-400" />
            )}
          </div>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed line-clamp-2">{method.description}</p>
      </div>
    </Card>
  )
}

export function Plans() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [allocatePlanId, setAllocatePlanId] = useState<number | null>(null)
  const [planStatuses, setPlanStatuses] = useState<Record<number, PlanStatus>>({})
  const [statusFilter, setStatusFilter] = useState<'all' | 'unallocated' | 'allocated' | 'completed'>('all')
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Enable real-time updates
  const { isConnected } = useRealtimeUpdates()

  const { data: allPlans = [], isLoading } = useQuery({
    queryKey: ['plans'],
    queryFn: plansApi.list,
  })

  // Check if we have plan statuses loaded
  const hasPlanStatuses = Object.keys(planStatuses).length > 0

  const getStatus = (planId: number) => planStatuses[planId]

  // Filter plans based on status - work with available data
  const plans = allPlans.filter(plan => {
    try {
      // Validate plan data
      if (!plan || typeof plan.plan_id !== 'number') {
        console.warn('Invalid plan data:', plan)
        return false
      }

      let shouldInclude = true

      if (statusFilter === 'all') {
        shouldInclude = true
      } else if (statusFilter === 'completed') {
        shouldInclude = plan.execution_status === 'completed'
      } else if (statusFilter === 'executing') {
        shouldInclude = plan.execution_status === 'executing'
      } else if (statusFilter === 'allocated') {
        const status = getStatus(plan.plan_id)
        const executionStatus = plan.execution_status || 'not_executed' // Default to not_executed if not set
        shouldInclude = status?.status === 'fully_allocated' && (executionStatus === 'not_executed' || executionStatus === 'executing')
      } else if (statusFilter === 'unallocated') {
        const status = getStatus(plan.plan_id)
        // If we don't have status data yet, assume unallocated for safety
        shouldInclude = !hasPlanStatuses || status?.status !== 'fully_allocated'
      } else {
        console.warn('Unknown statusFilter:', statusFilter)
        shouldInclude = true // Default to showing in unknown filter
      }

      return shouldInclude
    } catch (error) {
      console.error('Error filtering plan:', plan?.plan_id, error)
      return statusFilter === 'all' // Include in 'all' if filtering fails
    }
  })

  const { data: robots = [] } = useQuery({
    queryKey: ['robots'],
    queryFn: () => robotsApi.list(),
  })

  // Fetch robot health statuses for accurate execution readiness
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

  // Fetch allocation status for all plans
  useEffect(() => {
    const fetchStatuses = async () => {
      const statuses: Record<number, PlanStatus> = {}
      for (const plan of allPlans) {
        try {
          const status = await plansApi.getStatus(plan.plan_id)
          statuses[plan.plan_id] = status
        } catch (e) {
          // Ignore errors for individual plans
        }
      }
      setPlanStatuses(statuses)
    }
    if (allPlans.length > 0) {
      fetchStatuses()
    }
  }, [allPlans])

  const createPlanMutation = useMutation({
    mutationFn: plansApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      setIsModalOpen(false)
    },
  })

  const allocateMutation = useMutation({
    mutationFn: ({ planId, allocationStrategy }: { planId: number; allocationStrategy: string }) =>
      plansApi.allocate(planId, allocationStrategy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      setAllocatePlanId(null)
    },
  })

  const startMutation = useMutation({
    mutationFn: plansApi.start,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: plansApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
    },
  })

  const copyPlanMutation = useMutation({
    mutationFn: plansApi.copy,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
    },
  })

  // For now, use simple strategy selection
  const strategies = [
    { id: 'big_dag', name: 'Big DAG Planner', type: 'planning' },
    { id: 'llm_allocator', name: 'LLM Task Allocator', type: 'allocation' }
  ]

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-slate-700 rounded w-48"></div>
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-slate-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // Also show loading if we don't have plan statuses yet for filtering
  if (!hasPlanStatuses && allPlans.length > 0) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-slate-700 rounded w-48"></div>
          <div className="text-slate-400">Loading plan statuses...</div>
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-slate-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Plans</h1>
          <p className="text-slate-400 mt-1">
            Create and manage robot task plans for your goals
          </p>
        </div>
        <Button onClick={() => setIsModalOpen(true)}>
          <Plus className="w-4 h-4 mr-2" />
          New Plan
        </Button>
      </div>

      {/* Status Filter */}
      <div className="flex gap-2">
        {[
          { key: 'all', label: 'All Plans', count: allPlans.length },
          { key: 'unallocated', label: 'Unallocated', count: (() => {
            try {
              return allPlans.filter(p => {
                const status = getStatus(p.plan_id)
                return !hasPlanStatuses || status?.status !== 'fully_allocated'
              }).length
            } catch {
              return 0
            }
          })() },
          { key: 'allocated', label: 'Allocated', count: hasPlanStatuses ? (() => {
            try {
              return allPlans.filter(p => {
                const status = getStatus(p.plan_id)
                const executionStatus = p.execution_status || 'not_executed'
                return status?.status === 'fully_allocated' && (executionStatus === 'not_executed' || executionStatus === 'executing')
              }).length
            } catch {
              return 0
            }
          })() : 0 },
          { key: 'completed', label: 'Completed', count: allPlans.filter(p => p.execution_status === 'completed').length },
        ].map(({ key, label, count }) => (
          <button
            key={key}
            onClick={() => {
              console.log('Setting statusFilter to:', key, 'type:', typeof key)
              setStatusFilter(key as any)
            }}
            className={cn(
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              statusFilter === key
                ? 'bg-cyber-500 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            )}
          >
            {label} ({count})
          </button>
        ))}
      </div>

      {/* Plans Grid */}
      {plans.length === 0 ? (
        <EmptyState
          icon={<div>📋</div>}
          title="No plans found"
          description="No plans found. Try adjusting your filters or create a new plan."
          action={
            <Button onClick={() => setIsModalOpen(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Create Plan
            </Button>
          }
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {plans.map((plan) => {
            try {
              return (
                <Card key={plan.plan_id} className="p-4 hover:bg-slate-800/50 transition-colors cursor-pointer"
                      onClick={() => navigate(`/plans/${plan.plan_id}`)}>
              {/* Plan Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <h3 className="font-semibold text-white text-lg">Plan #{plan.plan_id}</h3>
                  <p className="text-sm text-slate-400">
                    {Array.isArray(plan.goal_ids) ? new Set(plan.goal_ids).size : 0} goals • {Array.isArray(plan.task_ids) ? plan.task_ids.length : 0} tasks
                  </p>
                </div>

                {/* Allocation Status Badge */}
                {(() => {
                  const status = getStatus(plan.plan_id)
                  if (!status) return null

                  // Determine badge based on allocation, execution status, and robot health
                  let badgeConfig;
                  const executionStatus = plan.execution_status || 'not_executed'

                  if (executionStatus === 'completed') {
                    badgeConfig = { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', label: 'Completed', icon: CheckCircle }
                  } else if (executionStatus === 'executing') {
                    badgeConfig = { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400', label: 'Running', icon: Play }
                  } else if (status.status === 'fully_allocated') {
                    // Check if robots are available for execution
                    // Use allocation artifacts to get robot assignments since plan.tasks isn't available in listing
                    let assignedRobotIds: string[] = []

                    if (plan.allocation_artifacts?.final_allocation?.allocations) {
                      // Extract robot IDs from final allocation
                      const allocations = plan.allocation_artifacts.final_allocation.allocations
                      assignedRobotIds = allocations
                        .map((allocation: any) => allocation.robot_id)
                        .filter((robotId: string) => robotId && robotId !== 'unassigned')
                    }

                    // Remove duplicates
                    assignedRobotIds = [...new Set(assignedRobotIds)]

                    const robotsReady = assignedRobotIds.length > 0 && assignedRobotIds.every((robotId) => {
                      const robot = robots?.find(r => r.robot_id === robotId)
                      const health = robotHealth ? robotHealth[robotId] : undefined
                      return robot && health?.reachable === true
                    })

                    if (robotsReady) {
                      badgeConfig = { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', label: 'Ready', icon: Zap }
                    } else {
                      badgeConfig = { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', label: 'Robots Offline', icon: AlertTriangle }
                    }
                  } else if (status.status === 'partially_allocated') {
                    badgeConfig = { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', label: 'Partial', icon: AlertCircle }
                  } else if (status.status === 'unallocated') {
                    badgeConfig = { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', label: 'Needs Allocation', icon: Users }
                  } else {
                    badgeConfig = { bg: 'bg-slate-500/10', border: 'border-slate-500/30', text: 'text-slate-400', label: 'No Tasks', icon: AlertTriangle }
                  }

                  const BadgeIcon = badgeConfig.icon
                  return (
                    <span className={cn('px-2 py-0.5 rounded text-xs flex items-center gap-1', badgeConfig.bg, 'border', badgeConfig.border, badgeConfig.text)}>
                      <BadgeIcon className="w-3 h-3" />
                      {badgeConfig.label}
                    </span>
                  )
                })()}
              </div>

              {/* Task allocation summary */}
              {(() => {
                const status = getStatus(plan.plan_id)
                return status && status.total_tasks > 0 ? (
                  <p className="text-sm text-slate-400 mb-3">
                    <span className={cn(
                      status.is_executable ? 'text-emerald-400' : 'text-amber-400'
                    )}>
                      {status.allocated_tasks}/{status.total_tasks} tasks allocated
                    </span>
                  </p>
                ) : null
              })()}

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                {/* Allocate Button */}
                {getStatus(plan.plan_id)?.status !== 'fully_allocated' && (
                  <Button
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      setAllocatePlanId(plan.plan_id)
                    }}
                    disabled={allocateMutation.isPending}
                  >
                    <Users className="w-3.5 h-3.5" />
                    Allocate
                  </Button>
                )}

                {/* Execute Button */}
                {getStatus(plan.plan_id)?.status === 'fully_allocated' &&
                 plan.execution_status !== 'completed' &&
                 plan.execution_status !== 'executing' && (
                  <Button
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      startMutation.mutate(plan.plan_id)
                    }}
                    disabled={startMutation.isPending || (() => {
                      // Check if ALL assigned robots are available and reachable
                      // Use allocation artifacts to get robot assignments
                      let assignedRobotIds: string[] = []

                      if (plan.allocation_artifacts?.final_allocation?.allocations) {
                        // Extract robot IDs from final allocation
                        const allocations = plan.allocation_artifacts.final_allocation.allocations
                        assignedRobotIds = allocations
                          .map((allocation: any) => allocation.robot_id)
                          .filter((robotId: string) => robotId && robotId !== 'unassigned')
                      }

                      // Remove duplicates
                      assignedRobotIds = [...new Set(assignedRobotIds)]

                      // Can't execute without robots
                      if (assignedRobotIds.length === 0) return true

                      // Must have ALL robots reachable
                      return !assignedRobotIds.every((robotId) => {
                        const robot = robots?.find(r => r.robot_id === robotId)
                        const health = robotHealth ? robotHealth[robotId] : undefined
                        return robot && health?.reachable === true
                      })
                    })()}
                  >
                    <Play className="w-3.5 h-3.5" />
                    Execute
                  </Button>
                )}

                <div className="flex items-center gap-2 ml-auto">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      copyPlanMutation.mutate(plan.plan_id)
                    }}
                    disabled={copyPlanMutation.isPending}
                    className="text-green-400 hover:text-green-300 hover:bg-green-500/10"
                    title="Copy plan for re-execution"
                  >
                    <Copy className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteMutation.mutate(plan.plan_id)
                    }}
                    disabled={deleteMutation.isPending}
                    className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
                </Card>
              )
            } catch (error) {
              console.error('Error rendering plan card:', plan.plan_id, error)
              return (
                <Card key={plan.plan_id} className="p-4 bg-red-900/20 border-red-500/30">
                  <div className="text-red-400">
                    Error loading plan #{plan.plan_id}
                  </div>
                </Card>
              )
            }
          })}
        </div>
      )}

      {/* Create Plan Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Create New Plan"
        size="wide"
        className="max-h-[90vh]"
      >
        <CreatePlanForm
          goals={goals}
          strategies={strategies}
          onSubmit={(data) => createPlanMutation.mutate(data)}
          onCancel={() => setIsModalOpen(false)}
          isLoading={createPlanMutation.isPending}
        />
      </Modal>

      {/* Allocate Plan Modal */}
      {allocatePlanId && (
        <Modal
          isOpen={!!allocatePlanId}
          onClose={() => setAllocatePlanId(null)}
          title={`Allocate Plan #${allocatePlanId}`}
          size="wide"
          className="max-h-[90vh]"
        >
          <AllocatePlanForm
            robots={robots}
            onSubmit={(allocationStrategy) =>
              allocateMutation.mutate({ planId: allocatePlanId, allocationStrategy })
            }
            onCancel={() => setAllocatePlanId(null)}
            isLoading={allocateMutation.isPending}
          />
        </Modal>
      )}
    </div>
  )
}

// =============================================================================
// Create Plan Form Component
// =============================================================================

interface CreatePlanFormProps {
  goals: Goal[]
  strategies: LocalStrategy[]
  onSubmit: (data: any) => void
  onCancel: () => void
  isLoading: boolean
}

function CreatePlanForm({ goals, strategies, onSubmit, onCancel, isLoading }: CreatePlanFormProps) {
  const [selectedGoals, setSelectedGoals] = useState<number[]>([])
  const [selectedPlanner, setSelectedPlanner] = useState<string>('')
  const [selectedAllocator, setSelectedAllocator] = useState<string>('')
  const [skipAllocation, setSkipAllocation] = useState<boolean>(false)
  const [plannerFilter, setPlannerFilter] = useState<string>('all')
  const [allocatorFilter, setAllocatorFilter] = useState<string>('all')
  const [goalSearch, setGoalSearch] = useState<string>('')

  // Load available planners and allocators dynamically
  const { data: planners = [] } = useQuery({
    queryKey: ['planners'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'planner')),
  })

  const { data: allocators = [] } = useQuery({
    queryKey: ['allocators'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'allocator')),
  })

  // Allocation options (only real allocators)
  const allocationOptions = [...allocators]

  // Filter planners and allocators by method type
  const filteredPlanners = planners.filter(p =>
    plannerFilter === 'all' || p.method_type === plannerFilter
  )
  const filteredAllocators = allocationOptions.filter(a =>
    allocatorFilter === 'all' || a.method_type === allocatorFilter
  )

  // Get unique method types for filters
  const plannerMethodTypes = ['all', ...new Set(planners.map(p => p.method_type))]
  const allocatorMethodTypes = ['all', ...new Set(allocationOptions.map(a => a.method_type))]

  // Filter and search goals
  const filteredGoals = goals.filter(goal =>
    goalSearch === '' ||
    goal.goal_id.toString().includes(goalSearch) ||
    goal.description.toLowerCase().includes(goalSearch.toLowerCase())
  )

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (selectedGoals.length === 0 || !selectedPlanner) return

    onSubmit({
      goal_ids: selectedGoals,
      planning_strategy: selectedPlanner,
      allocation_strategy: skipAllocation ? 'none' : (selectedAllocator || 'none'),
    })
  }

  return (
    <div className="space-y-6 max-h-[75vh] overflow-y-auto">
      {/* Planning and Allocation Methods - Two Column Layout */}
      <div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          {/* Planning Methods Column */}
          <div className="space-y-4">
            <h4 className="text-base font-medium text-white flex items-center gap-2">
              <span className="w-2 h-2 bg-cyber-400 rounded-full"></span>
              Planning Methods
            </h4>
            {/* Method Type Filter - Below header */}
            <div className="flex gap-1 flex-wrap">
              {plannerMethodTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => setPlannerFilter(type)}
                  className={cn(
                    'px-3 py-1 text-xs rounded-full border transition-all',
                    plannerFilter === type
                      ? 'bg-cyber-500/20 border-cyber-500/50 text-cyber-300'
                      : 'border-slate-600 text-slate-400 hover:border-slate-500'
                  )}
                >
                  {type === 'all' ? 'All' : type}
                </button>
              ))}
            </div>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {filteredPlanners.map((planner) => (
                <MethodSelectionCard
                  key={planner.type}
                  method={planner}
                  isSelected={selectedPlanner === planner.type}
                  onClick={() => setSelectedPlanner(planner.type)}
                />
              ))}
            </div>
          </div>

          {/* Allocation Methods Column */}
          <div className="space-y-4">
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
                      : 'border-slate-600 text-slate-400 hover:border-slate-500'
                  )}
                >
                  {type === 'all' ? 'All' : type}
                </button>
              ))}
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

            {/* No Allocation Option */}
            <div className="pt-4 border-t border-slate-700">
              <Card
                className={cn(
                  'cursor-pointer transition-all p-3',
                  skipAllocation
                    ? 'ring-2 ring-amber-500/50 border-amber-500 bg-gradient-to-br from-amber-500/10 to-orange-500/10'
                    : 'hover:border-slate-500 border-slate-700'
                )}
                onClick={() => {
                  setSkipAllocation(!skipAllocation)
                  if (!skipAllocation) {
                    setSelectedAllocator('') // Clear any selected allocator when skipping
                  }
                }}
              >
                <div className="flex items-center gap-3">
                  <div className={cn(
                    'relative w-5 h-5 rounded border-2 transition-all duration-200 flex items-center justify-center flex-shrink-0',
                    skipAllocation
                      ? 'bg-gradient-to-r from-amber-500 to-orange-500 border-transparent'
                      : 'border-slate-500'
                  )}>
                    {skipAllocation && (
                      <CheckCircle className="w-4 h-4 text-white" />
                    )}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-white text-sm mb-1">Skip Allocation</h4>
                    <p className="text-xs text-slate-400">Create plan without automatic robot allocation - assign robots manually later</p>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {/* Goals Selection - Beautiful Cards with Search */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white flex items-center gap-3">
            <span className="w-3 h-3 bg-gradient-to-r from-violet-400 to-pink-400 rounded-full"></span>
            Select Goals to Plan For
          </h3>
          <div className="text-sm text-slate-400">
            {selectedGoals.length} of {filteredGoals.length} selected
          </div>
        </div>

        {/* Search Bar */}
        <div className="relative mb-6">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search goals by ID or description..."
            value={goalSearch}
            onChange={(e) => setGoalSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:border-cyber-500 focus:ring-1 focus:ring-cyber-500 transition-all"
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
                  : 'hover:border-slate-500 hover:shadow-md hover:shadow-slate-500/10 border-slate-700'
              )}
              onClick={() => {
                if (selectedGoals.includes(goal.goal_id)) {
                  setSelectedGoals(selectedGoals.filter(id => id !== goal.goal_id))
                } else {
                  setSelectedGoals([...selectedGoals, goal.goal_id])
                }
              }}
            >
              <div className="flex items-start gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h4 className="font-semibold text-white text-sm">
                      Goal #{goal.goal_id}
                    </h4>
                  </div>
                  <p className="text-sm text-slate-300 leading-relaxed line-clamp-3">
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
          <div className="text-center py-8 text-slate-400">
            <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No goals match your search.</p>
          </div>
        )}
      </div>

      {/* Submit Section */}
      <div className="flex items-center justify-between pt-6 border-t border-slate-700">
        <div className="text-sm text-slate-400">
          {selectedGoals.length > 0 && selectedPlanner && (
            <span className="text-green-400">
              ✓ Ready to create plan with {selectedGoals.length} goal{selectedGoals.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
        <div className="flex gap-3">
          <Button
            type="button"
            variant="secondary"
            onClick={onCancel}
            disabled={isLoading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isLoading || selectedGoals.length === 0 || !selectedPlanner || (!selectedAllocator && !skipAllocation)}
            className="bg-gradient-to-r from-cyber-500 to-emerald-500 hover:from-cyber-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Creating Plan...' : `🚀 Create Plan (${selectedGoals.length})`}
          </Button>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Allocate Plan Form Component
// =============================================================================

interface AllocatePlanFormProps {
  robots: Robot[]
  onSubmit: (allocationStrategy: string) => void
  onCancel: () => void
  isLoading: boolean
}

function AllocatePlanForm({ robots, onSubmit, onCancel, isLoading }: AllocatePlanFormProps) {
  const [selectedAllocator, setSelectedAllocator] = useState<string>('')
  const [allocatorFilter, setAllocatorFilter] = useState<string>('all')

  // Load available allocators dynamically
  const { data: allocators = [] } = useQuery({
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
    onSubmit(selectedAllocator === 'none' ? 'none' : selectedAllocator)
  }

  return (
    <div className="space-y-6 max-h-[75vh] overflow-y-auto">
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
                      : 'border-slate-600 text-slate-400 hover:border-slate-500'
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
          <div className="text-sm text-slate-400">
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
                  : 'bg-slate-700/50 border-slate-600'
              )}
            >
              <div className="flex items-center gap-3">
                <div className={cn(
                  'w-3 h-3 rounded-full flex-shrink-0',
                  robot.status === 'running' || robot.status === 'registered'
                    ? 'bg-emerald-400 shadow-lg shadow-emerald-400/50'
                    : 'bg-slate-500'
                )} />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-white text-sm truncate">
                    {robot.robot_id}
                  </div>
                  <div className="text-xs text-slate-400">
                    {robot.robot_type}
                  </div>
                </div>
                <div className={cn(
                  'text-xs px-2 py-1 rounded-full',
                  robot.status === 'running' || robot.status === 'registered'
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-slate-600 text-slate-400'
                )}>
                  {robot.status === 'running' || robot.status === 'registered' ? 'Available' : 'Offline'}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Submit Section */}
      <div className="flex items-center justify-between pt-6 border-t border-slate-700">
        <div className="text-sm text-slate-400">
          {selectedAllocator && (
            <span className="text-green-400">
              ✓ Ready to allocate plan with {allocationOptions.find(a => a.type === selectedAllocator)?.name || 'selected method'}
            </span>
          )}
        </div>
        <div className="flex gap-3">
          <Button
            type="button"
            variant="secondary"
            onClick={onCancel}
            disabled={isLoading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isLoading || !selectedAllocator}
            className="bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Allocating Tasks...' : '🚀 Allocate Tasks'}
          </Button>
        </div>
      </div>
    </div>
  )
}
