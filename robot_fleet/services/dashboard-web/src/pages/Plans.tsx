import { useState, useEffect, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useLocation, useSearchParams } from 'react-router-dom'
import {
  Plus, Play, Trash2, CheckCircle, Activity,
  AlertCircle, Users, Zap, AlertTriangle, Copy, Search, Loader2, ChevronDown, Rocket, Wand2, FileText, XCircle, RotateCcw
} from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { ManualPlanCreation } from '../components/modals/ManualPlanCreation'
import { EmptyState } from '../components/common/EmptyState'
import { MethodDetailModal } from './Planners'
import { plansApi, goalsApi, robotsApi, methodsApi } from '../lib/api'
import { cn, getPlanningStrategyName, getAllocationStrategyName, getPlanningMethodId, getAllocationMethodId, setMethodData } from '../lib/utils'
import { useRealtimeUpdates } from '../lib/api'
import type { Robot, Goal, PlanStatus, PlanAllocationStatus } from '../types'
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
              method.method_type === 'manual' && 'bg-surface-elevated/30 border-border text-[var(--color-text-secondary)]'
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

export function Plans() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isManualPlanModalOpen, setIsManualPlanModalOpen] = useState(false)
  const [allocatePlanId, setAllocatePlanId] = useState<number | null>(null)
  const [copyingPlanId, setCopyingPlanId] = useState<number | null>(null)
  const [planStatuses, setPlanStatuses] = useState<Record<number, PlanStatus>>({})
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'unallocated' | 'allocated' | 'executing' | 'completed' | 'failed'>('all')
  const [sortBy, setSortBy] = useState<'id' | 'created_at'>('created_at')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [selectedMethod, setSelectedMethod] = useState<{ id: number; type: 'planner' | 'allocator' | null } | null>(null)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [searchParams] = useSearchParams()

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


  const { data: allPlans = [], isLoading } = useQuery({
    queryKey: ['plans'],
    queryFn: plansApi.list,
  })

  // Check if we have plan statuses loaded
  const hasPlanStatuses = Object.keys(planStatuses).length > 0

  const getStatus = (planId: number) => planStatuses[planId]

  // Filter and sort plans based on search, status, and sort criteria
  const plans = useMemo(() => {
    let filtered = allPlans

    // Apply search filter first
    if (searchTerm.trim()) {
      const searchLower = searchTerm.toLowerCase()
      filtered = filtered.filter(plan => {
        // Search in plan name, description, and ID
        const nameMatch = plan.name?.toLowerCase().includes(searchLower)
        const descMatch = plan.description?.toLowerCase().includes(searchLower)
        const idMatch = plan.plan_id.toString().includes(searchTerm)

        return nameMatch || descMatch || idMatch
      })
    }

    // Apply status filter
    filtered = filtered.filter(plan => {
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
        } else if (statusFilter === 'failed') {
          shouldInclude = plan.execution_status === 'failed'
        } else if (statusFilter === 'unallocated') {
          const status = getStatus(plan.plan_id)
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

    // Apply sorting
    return filtered.sort((a, b) => {
      let aValue: any, bValue: any

      if (sortBy === 'id') {
        aValue = a.plan_id
        bValue = b.plan_id
      } else if (sortBy === 'created_at') {
        aValue = new Date(a.created_at || 0)
        bValue = new Date(b.created_at || 0)
      } else {
        return 0
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : aValue < bValue ? -1 : 0
      } else {
        return aValue < bValue ? 1 : aValue > bValue ? -1 : 0
      }
    })
  }, [allPlans, searchTerm, statusFilter, sortBy, sortOrder, hasPlanStatuses])

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
          // Set a default status for failed requests to prevent UI errors
          statuses[plan.plan_id] = {
            plan_id: plan.plan_id,
            status: 'unknown' as PlanAllocationStatus,
            total_tasks: 0,
            allocated_tasks: 0,
            unallocated_task_ids: [],
            is_executable: false
          }
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
    mutationFn: ({ planId, name, description }: { planId: number; name: string; description: string }) =>
      plansApi.copy(planId, { name, description }),
    onMutate: ({ planId }) => {
      setCopyingPlanId(planId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['plans'] })
      setCopyingPlanId(null)
    },
    onError: () => {
      setCopyingPlanId(null)
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
          <div className="h-8 bg-surface-elevated rounded w-48"></div>
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-surface-elevated rounded"></div>
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
          <div className="h-8 bg-surface-elevated rounded w-48"></div>
          <div className="text-[var(--color-text-secondary)]">Loading plan statuses...</div>
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-surface-elevated rounded"></div>
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
          <p className="text-[var(--color-text-secondary)] mt-1">
            {plans.length} of {allPlans.length} plans
            {searchTerm && ` matching "${searchTerm}"`}
          </p>
        </div>
        <div className="flex gap-3">
          <Button onClick={() => setIsModalOpen(true)}>
            <Wand2 className="w-4 h-4 mr-2" />
            Create Plan with AI
          </Button>
          <Button onClick={() => setIsManualPlanModalOpen(true)} variant="secondary">
            <FileText className="w-4 h-4 mr-2" />
            Create Plan Manually
          </Button>
        </div>
      </div>

      {/* Search and Sort Controls */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search Input */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-[var(--color-text-secondary)]" />
          <input
            type="text"
            placeholder="Search plans by name, description, or ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-surface-overlay/50 border border-border rounded-lg text-white placeholder-[var(--color-text-muted)] focus:outline-none focus:border-cyber-500 focus:ring-1 focus:ring-cyber-500/50"
          />
        </div>

        {/* Sort Controls */}
        <div className="flex gap-2">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'id' | 'created_at')}
            className="px-3 py-2 bg-surface-overlay/50 border border-border rounded-lg text-white focus:outline-none focus:border-cyber-500 focus:ring-1 focus:ring-cyber-500/50"
          >
            <option value="created_at">Date Created</option>
            <option value="id">Plan ID</option>
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
          { key: 'executing', label: 'Executing', count: allPlans.filter(p => p.execution_status === 'executing').length },
          { key: 'completed', label: 'Completed', count: allPlans.filter(p => p.execution_status === 'completed').length },
          { key: 'failed', label: 'Failed', count: allPlans.filter(p => p.execution_status === 'failed').length },
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
                : 'bg-surface-elevated text-[var(--color-text)] hover:bg-surface-elevated'
            )}
          >
            {label} ({count})
          </button>
        ))}
      </div>

      

      {/* Plans Grid */}
      {plans.length === 0 ? (
        allPlans.length === 0 ? (
          <EmptyState
            icon={<div>📋</div>}
            title="No plans created"
            description="Create your first plan to define robot task execution sequences."
            action={
              <Button onClick={() => setIsModalOpen(true)}>
                <Plus className="w-4 h-4 mr-2" />
                Create Plan
              </Button>
            }
          />
        ) : (
          <div className="text-center py-12 text-[var(--color-text-muted)]">
            <Search className="w-12 h-12 text-[var(--color-text-muted)] mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium text-[var(--color-text-secondary)] mb-2">No plans match your search</p>
            <p className="text-[var(--color-text-muted)] mb-4">Try adjusting your search terms or clearing the filter</p>
            <Button
              variant="secondary"
              onClick={() => setSearchTerm('')}
              className="mr-2"
            >
              Clear Search
            </Button>
            <Button onClick={() => setStatusFilter('all')}>
              Show All Plans
            </Button>
          </div>
        )
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {plans.map((plan) => {
            // Safe rendering with fallbacks
            const status = getStatus(plan.plan_id)
            const executionStatus = plan.execution_status || 'not_executed'
            const goalCount = Array.isArray(plan.goal_ids) ? new Set(plan.goal_ids).size : 0
            const taskCount = Array.isArray(plan.task_ids) ? plan.task_ids.length : 0

            // Determine badge safely
            let badgeConfig = { bg: 'bg-surface-elevated/30', border: 'border-border', text: 'text-[var(--color-text-secondary)]', label: 'Unknown', icon: AlertTriangle }

            if (executionStatus === 'completed') {
              badgeConfig = { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', label: 'Completed', icon: CheckCircle }
            } else if (executionStatus === 'failed') {
              badgeConfig = { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', label: 'Failed', icon: XCircle }
            } else if (executionStatus === 'executing') {
              badgeConfig = { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', label: 'Running', icon: Play }
            } else if (status?.status === 'fully_allocated') {
              // Safe robot availability check
              let assignedRobotIds: string[] = []
              try {
                if (plan.allocation_artifacts?.final_allocation?.allocations) {
                  const allocations = plan.allocation_artifacts.final_allocation.allocations
                  assignedRobotIds = allocations
                    .map((allocation: any) => allocation?.robot_id)
                    .filter((robotId: string) => robotId && robotId !== 'unassigned') || []
                  assignedRobotIds = [...new Set(assignedRobotIds)]
                }
              } catch (e) {
                assignedRobotIds = []
              }

              const robotsReady = assignedRobotIds.length > 0 && assignedRobotIds.every((robotId) => {
                const robot = robots?.find(r => r.robot_id === robotId)
                const health = robotHealth ? robotHealth[robotId] : undefined
                return robot && health?.reachable === true
              })

              if (robotsReady) {
                badgeConfig = { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', label: 'Ready', icon: Zap }
              } else {
                badgeConfig = { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', label: 'Robots Offline', icon: AlertTriangle }
              }
            } else if (status?.status === 'partially_allocated') {
              badgeConfig = { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', label: 'Partial', icon: AlertCircle }
            } else if (status?.status === 'unallocated') {
              badgeConfig = { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', label: 'Needs Allocation', icon: Users }
            } else if (status?.status === 'empty' || (status?.total_tasks || 0) === 0) {
              badgeConfig = { bg: 'bg-surface-elevated/30', border: 'border-border', text: 'text-[var(--color-text-secondary)]', label: 'No Tasks', icon: AlertTriangle }
            }

            return (
              <Card key={plan.plan_id} className="p-4 hover:bg-surface-overlay/50 transition-colors cursor-pointer"
                    onClick={() => navigate(`/plans/${plan.plan_id}`)}>
                {/* Plan Header */}
                <div className="flex items-start justify-between mb-3">
                  {/* Plan ID Badge - Left */}
                  <span className="px-2 py-1 bg-yellow-500/20 border border-yellow-500/40 text-yellow-300 rounded text-xs font-semibold">
                    P{plan.plan_id}
                  </span>

                  {/* Action Buttons - Top Right */}
                  <div className="flex items-center gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        const copyName = plan.name ? `${plan.name} (Copy)` : `Plan #${plan.plan_id} (Copy)`
                        const copyDescription = plan.description || 'Copy of plan'

                        copyPlanMutation.mutate({
                          planId: plan.plan_id,
                          name: copyName,
                          description: copyDescription
                        })
                      }}
                      disabled={copyingPlanId !== null}
                      className="text-green-400 hover:text-green-300 hover:bg-green-500/10 p-2 h-8 w-8"
                      title="Copy plan for re-execution"
                    >
                      {copyingPlanId === plan.plan_id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteMutation.mutate(plan.plan_id)
                      }}
                      disabled={deleteMutation.isPending}
                      className="text-red-400 hover:text-red-300 hover:bg-red-500/10 p-2 h-8 w-8"
                      title="Delete plan"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                {/* Plan Name and Execution Status Row */}
                <div className="flex items-start justify-between mb-2">
                  {/* Plan Name and Description - Left */}
                  <div className="flex-1">
                    <h3 className="font-semibold text-white text-sm mb-1 line-clamp-1">
                      {plan.name}
                    </h3>
                    {plan.description && (
                      <p className="text-xs text-[var(--color-text-secondary)] line-clamp-2">
                        {plan.description}
                      </p>
                    )}
                  </div>

                  {/* Execution Status - Right */}
                  <span className={cn('px-2 py-1 rounded text-xs flex items-center gap-1 font-medium ml-2', badgeConfig.bg, 'border', badgeConfig.border, badgeConfig.text)}>
                    <badgeConfig.icon className="w-3 h-3" />
                    {badgeConfig.label}
                  </span>
                </div>

                {/* Strategy Information - Tag Style */}
                <div className="flex items-center gap-2 mb-4">
                  {plan.planning_strategy && (
                    <button
                      className="px-3 py-1.5 bg-blue-500/10 border border-blue-500/20 text-blue-400 rounded-lg text-sm font-semibold hover:bg-blue-500/30 transition-colors"
                      onClick={(e) => {
                        e.stopPropagation()
                        // Manual planning is not a method-details modal
                        if (plan.planning_strategy === 4) return
                        const methodId = getPlanningMethodId(plan.planning_strategy)
                        console.log('Planning method clicked:', plan.planning_strategy, '-> methodId:', methodId)
                        navigate(`/plans?plan=${plan.plan_id}&method_type=planner&method=${methodId}`)
                      }}
                    >
                      {getPlanningStrategyName(plan.planning_strategy)}
                    </button>
                  )}
                  {status?.status && status.status !== 'unallocated' && plan.allocation_strategy && plan.allocation_strategy !== 4 && (
                    <button
                      className="px-3 py-1.5 bg-purple-500/20 border border-purple-500/40 text-purple-300 rounded-lg text-sm font-semibold hover:bg-purple-500/30 transition-colors"
                      onClick={(e) => {
                        e.stopPropagation()
                        // Manual allocation is not a method-details modal
                        if (plan.allocation_strategy === 5) return
                        const methodId = getAllocationMethodId(plan.allocation_strategy)
                        navigate(`/plans?plan=${plan.plan_id}&method_type=allocator&method=${methodId}`)
                      }}
                    >
                      {getAllocationStrategyName(plan.allocation_strategy)}
                    </button>
                  )}
                </div>

                {/* Statistics Boxes - 3 Columns */}
                <div className="grid grid-cols-3 gap-2 mb-4">
                  {/* Goals Box */}
                  <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-2 text-center">
                    <div className="text-lg font-bold text-emerald-300 mb-1">
                      {goalCount}
                    </div>
                    <div className="text-xs text-emerald-400 font-medium">
                      Goals
                    </div>
                  </div>

                  {/* Tasks Box */}
                  <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2 text-center">
                    <div className="text-lg font-bold text-blue-300 mb-1">
                      {taskCount}
                    </div>
                    <div className="text-xs text-blue-400 font-medium">
                      Tasks
                    </div>
                  </div>

                  {/* Allocation Box */}
                  {status && status.total_tasks > 0 ? (
                    <div className="bg-surface-elevated/30 border border-border rounded p-2 text-center">
                      <div className="text-lg font-bold text-[var(--color-text)] mb-1">
                        {Math.round((status.allocated_tasks / status.total_tasks) * 100)}%
                      </div>
                      <div className="text-xs text-[var(--color-text-secondary)]">
                        Allocated
                      </div>
                    </div>
                  ) : (
                    <div className="bg-surface-elevated/30 border border-border rounded p-2 text-center opacity-50">
                      <div className="text-lg font-bold text-[var(--color-text-muted)] mb-1">
                        0%
                      </div>
                      <div className="text-xs text-[var(--color-text-muted)]">
                        Allocated
                      </div>
                    </div>
                  )}
                </div>

                {/* Action Buttons - Full Width */}
                <div className="w-full">
                  {/* Allocate Button */}
                  {status?.status !== 'fully_allocated' && (
                    <Button
                      className="w-full"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        setAllocatePlanId(plan.plan_id)
                      }}
                      disabled={allocateMutation.isPending}
                    >
                      <Users className="w-3.5 h-3.5 mr-2" />
                      Allocate Tasks to Robots
                    </Button>
                  )}

                  {/* Execute Button */}
                  {getStatus(plan.plan_id)?.status === 'fully_allocated' &&
                   plan.execution_status !== 'completed' &&
                   plan.execution_status !== 'executing' &&
                   plan.execution_status !== 'failed' && (
                    <Button
                      className="w-full"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        navigate(`/plans/${plan.plan_id}/execute`)
                      }}
                    >
                      <Play className="w-3.5 h-3.5 mr-2" />
                      Execute Plan
                    </Button>
                  )}

                  {/* View Execution (for running, completed, or failed plans) */}
                  {(plan.execution_status === 'executing' || plan.execution_status === 'completed' || plan.execution_status === 'failed') && (
                    <Button
                      className="w-full"
                      size="sm"
                      variant="secondary"
                      onClick={(e) => {
                        e.stopPropagation()
                        navigate(`/plans/${plan.plan_id}/execute`)
                      }}
                    >
                      <Activity className="w-3.5 h-3.5 mr-2" />
                      View Execution
                    </Button>
                  )}

                  {/* Restart Button (for failed plans) */}
                  {plan.execution_status === 'failed' && (
                    <Button
                      className="w-full mt-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/30"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        const retryName = plan.name ? `${plan.name} (Retry)` : `Plan #${plan.plan_id} (Retry)`
                        const retryDescription = plan.description || 'Retry of failed plan'
                        copyPlanMutation.mutate({
                          planId: plan.plan_id,
                          name: retryName,
                          description: retryDescription,
                        })
                      }}
                      disabled={copyingPlanId !== null}
                    >
                      <RotateCcw className="w-3.5 h-3.5 mr-2" />
                      Copy & Retry
                    </Button>
                  )}
                </div>
              </Card>
              )
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

      {/* Method Detail Modal */}
      <MethodDetailModal
        methodId={selectedMethod?.id || null}
        methodType={selectedMethod?.type || undefined}
        isOpen={!!selectedMethod}
        onClose={() => {
          setSelectedMethod(null)
          navigate('/plans', { replace: true })
        }}
      />

      {/* Copying Progress Modal */}
      {copyingPlanId && (
        <Modal isOpen={true} onClose={() => {}} title="" size="sm">
          <div className="flex flex-col items-center justify-center py-8">
            <Loader2 className="w-8 h-8 animate-spin text-emerald-400 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">Copying Plan #{copyingPlanId}</h3>
            <p className="text-[var(--color-text-secondary)] text-center">
              Creating a duplicate plan for re-execution...
            </p>
          </div>
        </Modal>
      )}

      {/* Manual Plan Creation Modal */}
      <ManualPlanCreation
        isOpen={isManualPlanModalOpen}
        onClose={() => setIsManualPlanModalOpen(false)}
      />

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
  const [selectedPlanner, setSelectedPlanner] = useState<number | null>(null)
  const [selectedAllocator, setSelectedAllocator] = useState<number | null>(null)
  const [skipAllocation, setSkipAllocation] = useState<boolean>(false)
  const [plannerFilter, setPlannerFilter] = useState<string>('all')
  const [allocatorFilter, setAllocatorFilter] = useState<string>('all')
  const [goalSearch, setGoalSearch] = useState<string>('')
  const [planName, setPlanName] = useState<string>('')
  const [planDescription, setPlanDescription] = useState<string>('')

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

  // Update method data for name lookups
  useEffect(() => {
    setMethodData(planners, allocators)
  }, [planners, allocators])

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
    if (selectedGoals.length === 0 || selectedPlanner === null || !planName.trim() || !planDescription.trim()) return

    onSubmit({
      goal_ids: selectedGoals,
      planning_strategy: selectedPlanner!,
      allocation_strategy: skipAllocation ? 4 : (selectedAllocator || 4), // 4 = NONE
      name: planName.trim(),
      description: planDescription.trim(),
    })
  }

  return (
    <div className="space-y-6 max-h-[75vh] overflow-y-auto">
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
                      : 'border-border text-[var(--color-text-secondary)] hover:border-border-strong'
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
                  isSelected={selectedPlanner === planner.id}
                  onClick={() => setSelectedPlanner(planner.id)}
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
                      : 'border-border text-[var(--color-text-secondary)] hover:border-border-strong'
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
                  isSelected={selectedAllocator === allocator.id}
                  onClick={() => setSelectedAllocator(allocator.id)}
                />
              ))}
            </div>

            {/* No Allocation Option */}
            <div className="pt-4 border-t border-border">
              <Card
                className={cn(
                  'cursor-pointer transition-all p-3',
                  skipAllocation
                    ? 'ring-2 ring-amber-500/50 border-amber-500 bg-gradient-to-br from-amber-500/10 to-orange-500/10'
                    : 'hover:border-border-strong border-border'
                )}
                onClick={() => {
                  setSkipAllocation(!skipAllocation)
                  if (!skipAllocation) {
                    setSelectedAllocator(null) // Clear any selected allocator when skipping
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
                    <p className="text-xs text-[var(--color-text-secondary)]">Create plan without automatic robot allocation - assign robots manually later</p>
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
                  : 'hover:border-border-strong hover:shadow-md hover:shadow-slate-500/10 border-border'
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

      {/* Submit Section */}
      <div className="flex items-center justify-between pt-6 border-t border-border">
        <div className="text-sm text-[var(--color-text-secondary)]">
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
            disabled={isLoading || selectedGoals.length === 0 || selectedPlanner === null || (!selectedAllocator && !skipAllocation) || !planName.trim() || !planDescription.trim()}
            title={
              isLoading ? 'Creating plan...' :
              selectedGoals.length === 0 ? 'Select at least one goal' :
              selectedPlanner === null ? 'Select a planning method' :
              (!selectedAllocator && !skipAllocation) ? 'Select an allocation method or skip allocation' :
              !planName.trim() ? 'Enter a plan name' :
              !planDescription.trim() ? 'Enter a plan description' :
              'Ready to create plan'
            }
            className="bg-gradient-to-r from-cyber-500 to-emerald-500 hover:from-cyber-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Creating Plan...' : (
              <>
                <Rocket className="w-4 h-4 mr-2" />
                Create Plan ({selectedGoals.length})
              </>
            )}
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
                  : 'bg-surface-elevated/50 border-border'
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

      {/* Submit Section */}
      <div className="flex items-center justify-between pt-6 border-t border-border">
        <div className="text-sm text-[var(--color-text-secondary)]">
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
            {isLoading ? 'Allocating Tasks...' : (
              <>
                <Rocket className="w-4 h-4 mr-2" />
                Allocate Tasks
              </>
            )}
          </Button>
        </div>
      </div>
      </>
      )}

    </div>
  )
}
