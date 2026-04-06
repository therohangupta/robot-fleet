import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Bot, Plus, Trash2, Wifi, WifiOff, RefreshCw, CheckCircle, XCircle, Loader2, FileCode, Download, Settings, Cpu, Network, FileX, Target, Wrench, Info, Send, RotateCcw, Video, Activity, Terminal } from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { useRealtimeUpdates, robotsApi } from '../lib/api'

import { cn } from '../lib/utils'
import type { Robot } from '../types'

// =============================================================================
// Health Check Types
// =============================================================================

interface RobotHealth {
  robot_id: string
  host: string
  port: number
  reachable: boolean
  latency_ms?: number
  error?: string
}

interface RobotYamlDetails {
  robot: Robot
  yaml_path: string | null
  yaml_content: Record<string, unknown> | null
}

// =============================================================================
// Robot Allocations API
// =============================================================================

interface RobotAllocations {
  robot_id: string
  plans_count: number
  goals_count: number
  tasks_count: number
  plans: Array<{
    plan_id: number
    goal_ids: number[]
    task_count: number
    status: string
    name: string
    description: string
  }>
  goals: number[]
}


// Info Card Component for technical details
function InfoCard({
  title,
  value,
  subtitle,
  icon,
  status
}: {
  title: string
  value: string
  subtitle?: string
  icon: React.ReactNode
  status?: boolean
}) {
  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <div className={cn(
          "p-2 rounded-lg flex-shrink-0",
          status === true ? "bg-emerald-500/10 text-emerald-400" :
          status === false ? "bg-red-500/10 text-red-400" :
          "bg-surface-overlay text-[var(--color-text-secondary)]"
        )}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-[var(--color-text-muted)] uppercase tracking-wide font-medium">{title}</p>
          <p className="font-medium text-white truncate">{value}</p>
          {subtitle && <p className="text-xs text-[var(--color-text-secondary)] truncate">{subtitle}</p>}
        </div>
      </div>
    </Card>
  )
}

// =============================================================================
// Robot Card with Health Status
// =============================================================================

function RobotCard({
  robot,
  health,
  allocations,
  onClick,
  onDelete,
  onCheckHealth,
  isChecking
}: {
  robot: Robot
  health?: RobotHealth
  allocations?: { plans_count: number; goals_count: number; tasks_count: number }
  onClick: () => void
  onDelete: (id: string) => void
  onCheckHealth: (id: string) => void
  isChecking: boolean
}) {
  const isReachable = health?.reachable === true

  return (
    <Card hover className="relative overflow-hidden cursor-pointer" onClick={onClick}>
      <div className={cn(
        'absolute top-0 left-0 w-1 h-full',
        isChecking ? 'bg-yellow-500' : isReachable ? 'bg-emerald-500' : 'bg-red-500'
      )} />

      <div className="pl-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={cn(
              'w-10 h-10 rounded-lg flex items-center justify-center',
              isReachable ? 'bg-emerald-500/10' : 'bg-surface-overlay'
            )}>
              <Bot className={cn('w-5 h-5', isReachable ? 'text-emerald-400' : 'text-[var(--color-text-muted)]')} />
            </div>
            <div>
              <h3 className="font-semibold text-white">{robot.robot_id}</h3>
              <p className="text-xs text-[var(--color-text-muted)]">{robot.robot_type}</p>
            </div>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-1.5">
            {isChecking ? (
              <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />
            ) : isReachable ? (
              <CheckCircle className="w-4 h-4 text-emerald-400" />
            ) : (
              <XCircle className="w-4 h-4 text-red-400" />
            )}
            <span className={cn(
              'text-xs font-medium',
              isChecking ? 'text-yellow-400' : isReachable ? 'text-emerald-400' : 'text-red-400'
            )}>
              {isChecking ? 'Checking...' : isReachable ? 'Connected' : 'Unreachable'}
            </span>
          </div>
        </div>

        {/* Allocations */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          <span className="px-2 py-0.5 bg-surface-overlay rounded text-xs text-[var(--color-text-secondary)] font-mono">
            {allocations?.plans_count || 0} plans
          </span>
          <span className="px-2 py-0.5 bg-surface-overlay rounded text-xs text-[var(--color-text-secondary)] font-mono">
            {allocations?.goals_count || 0} goals
          </span>
          <span className="px-2 py-0.5 bg-surface-overlay rounded text-xs text-[var(--color-text-secondary)] font-mono">
            {allocations?.tasks_count || 0} tasks
          </span>
        </div>

        {/* Connection Info & Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-border">
          <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
            {isReachable ? (
              <Wifi className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <WifiOff className="w-3.5 h-3.5 text-red-400" />
            )}
            <span className="font-mono">
              {robot.task_server_info?.host}:{robot.task_server_info?.port}
            </span>
            {health?.latency_ms && (
              <span className="text-emerald-400">({Math.round(health.latency_ms)}ms)</span>
            )}
          </div>
          <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onCheckHealth(robot.robot_id)}
              className="text-[var(--color-text-secondary)] hover:text-white"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onDelete(robot.robot_id)}
              className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Error message */}
        {health?.error && (
          <p className="text-xs text-red-400 mt-2">{health.error}</p>
        )}
      </div>
    </Card>
  )
}

// Config Section Component
function ConfigSection({
  title,
  items,
  icon
}: {
  title: string
  items: Array<{key: string, value: any}>
  icon?: React.ReactNode
}) {
  const [isExpanded, setIsExpanded] = useState(true)

  return (
    <Card className="p-4">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between w-full text-left mb-3"
      >
        <div className="flex items-center gap-2">
          {icon && <div className="text-[var(--color-text-secondary)]">{icon}</div>}
          <h5 className="font-medium text-white">{title}</h5>
          <span className="text-xs text-[var(--color-text-muted)] bg-surface-overlay px-2 py-0.5 rounded-full">
            {items.length}
          </span>
        </div>
        <div className="text-[var(--color-text-secondary)]">
          {isExpanded ? '−' : '+'}
        </div>
      </button>

      {isExpanded && (
        <div className="space-y-2">
          {items.map(({key, value}) => (
            <div key={key} className="flex justify-between items-start py-1 border-b border-border/50">
              <span className="text-sm text-[var(--color-text-secondary)] font-mono flex-shrink-0 mr-4">{key}:</span>
              <span className="text-sm text-[var(--color-text)] font-mono break-all text-right">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

// Configuration Viewer Component
function ConfigurationViewer({
  config,
  filePath
}: {
  config: Record<string, unknown>
  filePath: string
}) {
  // Parse configuration into logical sections
  const parseConfigSections = (config: Record<string, unknown>) => {
    const sections = []

    // Robot metadata
    const robotSection = []
    if (config.robot_id) robotSection.push({key: 'robot_id', value: config.robot_id})
    if (config.robot_type) robotSection.push({key: 'robot_type', value: config.robot_type})
    if (config.description) robotSection.push({key: 'description', value: config.description})
    if (robotSection.length > 0) {
      sections.push({
        title: 'Robot Identity',
        items: robotSection,
        icon: <Bot className="w-4 h-4" />
      })
    }

    // Connection settings
    const connectionSection = []
    if (config.host) connectionSection.push({key: 'host', value: config.host})
    if (config.port) connectionSection.push({key: 'port', value: config.port})
    if (connectionSection.length > 0) {
      sections.push({
        title: 'Network Configuration',
        items: connectionSection,
        icon: <Network className="w-4 h-4" />
      })
    }

    // Hardware specs
    const hardwareSection = []
    if (config.hardware) hardwareSection.push({key: 'hardware', value: config.hardware})
    if (config.capabilities) hardwareSection.push({key: 'capabilities', value: config.capabilities})
    if (config.sensors) hardwareSection.push({key: 'sensors', value: config.sensors})
    if (hardwareSection.length > 0) {
      sections.push({
        title: 'Hardware Specifications',
        items: hardwareSection,
        icon: <Cpu className="w-4 h-4" />
      })
    }

    // Software settings
    const softwareSection = []
    if (config.software) softwareSection.push({key: 'software', value: config.software})
    if (config.parameters) softwareSection.push({key: 'parameters', value: config.parameters})
    if (config.calibration) softwareSection.push({key: 'calibration', value: config.calibration})
    if (softwareSection.length > 0) {
      sections.push({
        title: 'Software Configuration',
        items: softwareSection,
        icon: <Settings className="w-4 h-4" />
      })
    }

    // Everything else
    const otherItems = Object.entries(config).filter(([key]) =>
      !['robot_id', 'robot_type', 'description', 'host', 'port', 'hardware', 'capabilities', 'sensors', 'software', 'parameters', 'calibration'].includes(key)
    ).map(([key, value]) => ({key, value}))

    if (otherItems.length > 0) {
      sections.push({
        title: 'Additional Settings',
        items: otherItems,
        icon: <FileCode className="w-4 h-4" />
      })
    }

    return sections
  }

  const sections = parseConfigSections(config)

  return (
    <div className="space-y-4">
      <div className="text-xs text-[var(--color-text-secondary)] font-mono p-2 bg-surface/60 rounded border border-border">
        📁 {filePath}
      </div>

      {sections.map(section => (
        <ConfigSection
          key={section.title}
          title={section.title}
          items={section.items}
          icon={section.icon}
        />
      ))}

      {/* Raw JSON fallback */}
      <Card className="p-4">
        <h5 className="font-medium text-white mb-3 flex items-center gap-2">
          <FileCode className="w-4 h-4" />
          Raw Configuration
        </h5>
        <pre className="p-3 bg-surface rounded text-xs text-[var(--color-text)] overflow-auto max-h-48 font-mono">
          {JSON.stringify(config, null, 2)}
        </pre>
      </Card>
    </div>
  )
}

// =============================================================================
// Robot Detail Modal
// =============================================================================

// =============================================================================
// Tab Components
// =============================================================================

function TabNavigation({
  tabs,
  activeTab,
  onTabChange
}: {
  tabs: Array<{id: string, label: string, icon: React.ReactNode}>
  activeTab: string
  onTabChange: (tabId: string) => void
}) {
  return (
    <div className="flex border-b border-border mb-6">
      {tabs.map(tab => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={cn(
            'flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors',
            activeTab === tab.id
              ? 'border-cyber-500 text-cyber-400'
              : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
          )}
        >
          {tab.icon}
          {tab.label}
        </button>
      ))}
    </div>
  )
}

function OverviewTab({ robot, health }: { robot: Robot, health?: RobotHealth }) {
  const isReachable = health?.reachable ?? false

  return (
    <div className="space-y-6">
      {/* Technical Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <InfoCard
          title="Embodiment"
          value={robot.robot_type}
          icon={<Bot className="w-5 h-5" />}
        />
        <InfoCard
          title="Connection"
          value={health?.reachable ? "Online" : "Offline"}
          subtitle={`${robot.task_server_info?.host || 'Unknown'}:${robot.task_server_info?.port || 'Unknown'}`}
          icon={<Network className="w-5 h-5" />}
          status={health?.reachable}
        />
        <InfoCard
          title="Capabilities"
          value={`${robot.capabilities.length} Skills`}
          subtitle="Available Actions"
          icon={<Settings className="w-5 h-5" />}
        />
      </div>

      {/* Description */}
      {robot.description && (
        <Card className="p-4">
          <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-2 uppercase tracking-wide">Description</h4>
          <p className="text-white leading-relaxed">{robot.description}</p>
        </Card>
      )}

      {/* Connection Details */}
      <Card className="p-4">
        <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3 uppercase tracking-wide">Connection Details</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-[var(--color-text-muted)]">Host:</span>
            <span className="text-white ml-2 font-mono">{robot.task_server_info?.host || 'Unknown'}</span>
          </div>
          <div>
            <span className="text-[var(--color-text-muted)]">Port:</span>
            <span className="text-white ml-2 font-mono">{robot.task_server_info?.port || 'Unknown'}</span>
          </div>
          <div>
            <span className="text-[var(--color-text-muted)]">Status:</span>
            <span className={cn("ml-2", isReachable ? "text-emerald-400" : "text-red-400")}>
              {isReachable ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          {health?.latency_ms && (
            <div>
              <span className="text-[var(--color-text-muted)]">Latency:</span>
              <span className="text-emerald-400 ml-2">{Math.round(health.latency_ms)}ms</span>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}

function AllocationsTab({ robotId }: { robotId: string }) {
  const navigate = useNavigate()
  const [statusFilter, setStatusFilter] = useState<string>('all')

  const { data: allocations, isLoading } = useQuery({
    queryKey: ['robot-allocations', robotId],
    queryFn: () => robotsApi.getAllocations(robotId),
  })

  const filteredPlans = allocations?.plans.filter(plan => {
    if (statusFilter === 'all') return true
    return plan.status === statusFilter
  }) || []

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-[var(--color-text-secondary)]" />
        <span className="ml-3 text-[var(--color-text-secondary)]">Loading allocations...</span>
      </div>
    )
  }

  if (!allocations) {
    return (
      <div className="text-center py-12 text-[var(--color-text-muted)]">
        <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p>Failed to load allocation data</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Allocation Summary */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-white">{allocations.plans_count}</div>
          <div className="text-sm text-[var(--color-text-secondary)]">Active Plans</div>
        </Card>
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-white">{allocations.goals_count}</div>
          <div className="text-sm text-[var(--color-text-secondary)]">Goals Assigned</div>
        </Card>
        <Card className="p-4 text-center">
          <div className="text-2xl font-bold text-white">{allocations.tasks_count}</div>
          <div className="text-sm text-[var(--color-text-secondary)]">Tasks Assigned</div>
        </Card>
      </div>

      {/* Plans Section */}
      {allocations.plans.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-white flex items-center gap-2">
              <Target className="w-4 h-4" />
              Allocated Plans ({allocations.plans.length})
            </h4>
          </div>

          {/* Status Filters */}
          <div className="flex gap-2">
            {[
              { key: 'all', label: 'All Plans', count: allocations.plans.length },
              { key: 'not_executed', label: 'Pending', count: allocations.plans.filter(p => p.status === 'not_executed').length },
              { key: 'executing', label: 'Executing', count: allocations.plans.filter(p => p.status === 'executing').length },
              { key: 'completed', label: 'Completed', count: allocations.plans.filter(p => p.status === 'completed').length },
            ].map(({ key, label, count }) => (
              <Button
                key={key}
                variant={statusFilter === key ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => setStatusFilter(key)}
                className="text-xs"
              >
                {label} ({count})
              </Button>
            ))}
          </div>

          {/* Plans Grid */}
          <div className="max-h-96 overflow-y-auto">
            {filteredPlans.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                {filteredPlans.map((plan) => (
                  <Card
                    key={plan.plan_id}
                    className="p-3 cursor-pointer hover:bg-surface-elevated/50 transition-all duration-200 hover:shadow-lg hover:scale-[1.02]"
                    onClick={() => navigate(`/plans/${plan.plan_id}?from=robot-${robotId}`)}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <div className="font-semibold text-white text-sm">#{plan.plan_id}: {plan.name}</div>
                      <div className={cn(
                        'px-1.5 py-0.5 rounded text-xs font-medium',
                        plan.status === 'completed' ? 'bg-emerald-500/20 text-emerald-300' :
                        plan.status === 'executing' ? 'bg-amber-500/20 text-amber-300' :
                        plan.status === 'failed' ? 'bg-red-500/20 text-red-300' :
                        'bg-amber-500/20 text-amber-300'
                      )}>
                        {plan.status === 'not_executed' ? 'Pending' :
                         plan.status === 'executing' ? 'Running' :
                         plan.status.charAt(0).toUpperCase() + plan.status.slice(1)}
                      </div>
                    </div>

                    {/* Plan Name and Description */}
                    <div className="mb-3">
                      {plan.description && (
                        <p className="text-xs text-[var(--color-text-secondary)] line-clamp-2">
                          {plan.description}
                        </p>
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      {/* Robot Tasks Box */}
                      <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2 text-center">
                        <div className="text-lg font-bold text-blue-300 mb-1">
                          {plan.task_count}
                        </div>
                        <div className="text-xs text-blue-400 font-medium leading-tight">
                          Tasks Assigned<br />to this Robot
                        </div>
                      </div>

                      {/* Plan Goals Box */}
                      <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-2 text-center">
                        <div className="text-lg font-bold text-emerald-300 mb-1">
                          {plan.goal_ids.length}
                        </div>
                        <div className="text-xs text-emerald-400 font-medium">
                          Plan Goals
                        </div>
                      </div>
                    </div>

                    {plan.goal_ids.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-border">
                        <div className="flex flex-wrap gap-0.5">
                          {plan.goal_ids.slice(0, 4).map((goalId) => (
                            <span
                              key={goalId}
                              className="px-1.5 py-0.5 bg-cyber-500/20 text-cyber-300 rounded text-xs font-medium"
                            >
                              G{goalId}
                            </span>
                          ))}
                          {plan.goal_ids.length > 4 && (
                            <span className="px-1.5 py-0.5 bg-surface-elevated text-[var(--color-text-secondary)] rounded text-xs">
                              +{plan.goal_ids.length - 4}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-[var(--color-text-muted)]">
                <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No plans match the selected filter</p>
              </div>
            )}
          </div>
        </div>
      )}

      {allocations.plans.length === 0 && (
        <div className="text-center py-12 text-[var(--color-text-muted)]">
          <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p>This robot has no active plan allocations</p>
        </div>
      )}
    </div>
  )
}

function CapabilitiesTab({ capabilities }: { capabilities: string[] }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Wrench className="w-5 h-5 text-[var(--color-text-secondary)]" />
          <h4 className="text-lg font-semibold text-white">Robot Capabilities</h4>
        </div>
        <span className="text-sm text-[var(--color-text-muted)] bg-surface-overlay px-3 py-1 rounded-full">
          {capabilities.length} skills
        </span>
      </div>

      <Card className="p-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {capabilities.map((capability, index) => (
            <div
              key={index}
              className="flex items-center gap-3 p-3 bg-surface-overlay/50 rounded-lg border border-border"
            >
              <div className="w-2 h-2 bg-cyber-400 rounded-full flex-shrink-0"></div>
              <span className="text-sm text-[var(--color-text)] font-mono">
                {capability}
              </span>
            </div>
          ))}
        </div>
        {capabilities.length === 0 && (
          <div className="text-center py-8 text-[var(--color-text-muted)]">
            <Wrench className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No capabilities configured</p>
          </div>
        )}
      </Card>
    </div>
  )
}

function ConfigurationTab({ robot }: { robot: Robot }) {
  const [yamlDetails, setYamlDetails] = useState<RobotYamlDetails | null>(null)
  const [isLoadingYaml, setIsLoadingYaml] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null)
  const queryClient = useQueryClient()

  // Fetch YAML details
  useEffect(() => {
    if (robot) {
      setIsLoadingYaml(true)
      fetch(`/api/robots/${robot.robot_id}/yaml`)
        .then(res => res.json())
        .then(data => {
          setYamlDetails(data)
          setIsLoadingYaml(false)
        })
        .catch(() => setIsLoadingYaml(false))
    }
  }, [robot])

  const handleRefreshFromYaml = async () => {
    if (!robot) return
    setIsRefreshing(true)
    setRefreshMessage(null)

    try {
      const response = await fetch(`/api/robots/${robot.robot_id}/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: yamlDetails?.yaml_path })
      })

      if (response.ok) {
        const data = await response.json()
        setRefreshMessage('✓ Robot refreshed from YAML successfully!')
        queryClient.invalidateQueries({ queryKey: ['robots'] })
        // Re-fetch YAML details
        const yamlRes = await fetch(`/api/robots/${robot.robot_id}/yaml`)
        const yamlData = await yamlRes.json()
        setYamlDetails(yamlData)
      } else {
        const error = await response.json()
        setRefreshMessage(`✗ ${error.detail || 'Failed to refresh'}`)
      }
    } catch (error) {
      setRefreshMessage('✗ Failed to refresh from YAML')
    }
    setIsRefreshing(false)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileCode className="w-5 h-5 text-[var(--color-text-secondary)]" />
          <h4 className="text-lg font-semibold text-white">Configuration Details</h4>
        </div>
        <Button
          size="sm"
          variant="secondary"
          onClick={handleRefreshFromYaml}
          disabled={isRefreshing || !yamlDetails?.yaml_path}
        >
          <Download className={cn("w-3.5 h-3.5", isRefreshing && "animate-spin")} />
          {isRefreshing ? 'Refreshing...' : 'Refresh from YAML'}
        </Button>
      </div>

      {isLoadingYaml ? (
        <Card className="p-8">
          <div className="flex items-center justify-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-[var(--color-text-secondary)]" />
            <span className="text-[var(--color-text-secondary)]">Loading configuration details...</span>
          </div>
        </Card>
      ) : yamlDetails?.yaml_content ? (
        <ConfigurationViewer
          config={yamlDetails.yaml_content}
          filePath={yamlDetails.yaml_path || 'Unknown path'}
        />
      ) : (
        <Card className="p-8">
          <div className="flex flex-col items-center justify-center gap-3">
            <FileX className="w-8 h-8 text-[var(--color-text-muted)]" />
            <div className="text-center">
              <h5 className="font-medium text-[var(--color-text-secondary)] mb-1">Configuration Not Found</h5>
              <p className="text-sm text-[var(--color-text-muted)]">YAML source file not available for this robot</p>
            </div>
          </div>
        </Card>
      )}

      {refreshMessage && (
        <div className={cn(
          'p-3 rounded-lg border text-sm font-medium',
          refreshMessage.startsWith('✓')
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            : 'bg-red-500/10 border-red-500/20 text-red-400'
        )}>
          {refreshMessage}
        </div>
      )}
    </div>
  )
}

function RobotDetailModal({
  robot,
  health,
  isOpen,
  onClose,
  onRefresh,
  activeTab,
  onTabChange
}: {
  robot: Robot | null
  health?: RobotHealth
  isOpen: boolean
  onClose: () => void
  onRefresh: () => void
  activeTab: string
  onTabChange: (tab: string) => void
}) {

  const tabs = [
    { id: 'overview', label: 'Overview', icon: <Info className="w-4 h-4" /> },
    { id: 'allocations', label: 'Allocations', icon: <Target className="w-4 h-4" /> },
    { id: 'capabilities', label: 'Capabilities', icon: <Wrench className="w-4 h-4" /> },
    { id: 'config', label: 'Configuration', icon: <FileCode className="w-4 h-4" /> },
    { id: 'send-task', label: 'Send Task', icon: <Send className="w-4 h-4" /> }
  ]

  if (!robot) return null

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`${robot.robot_id}`} size="wide" className="max-h-[90vh]">
      <div className="space-y-6 max-h-[75vh] overflow-y-auto">
        {/* Tab Navigation */}
        <TabNavigation tabs={tabs} activeTab={activeTab} onTabChange={onTabChange} />

        {/* Tab Content */}
        {activeTab === 'overview' && <OverviewTab robot={robot} health={health} />}
        {activeTab === 'allocations' && <AllocationsTab robotId={robot.robot_id} />}
        {activeTab === 'capabilities' && <CapabilitiesTab capabilities={robot.capabilities} />}
        {activeTab === 'config' && <ConfigurationTab robot={robot} />}
        {activeTab === 'send-task' && <SendTaskTab robot={robot} health={health} />}

      </div>
    </Modal>
  )
}

// =============================================================================
// Send Task Tab Component
// =============================================================================

function SendTaskTab({ robot, health }: { robot: Robot, health?: RobotHealth }) {
  const [taskDescription, setTaskDescription] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [response, setResponse] = useState<{
    success: boolean
    message: string
    replan?: boolean
    timestamp: Date
  } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSendTask = async () => {
    if (!taskDescription.trim()) {
      setError('Please enter a task description')
      return
    }

    if (!robot.task_server_info?.host || !robot.task_server_info?.port) {
      setError('Robot server information not available')
      return
    }

    setIsSending(true)
    setError(null)
    setResponse(null)

    try {
      const host = robot.task_server_info.host === 'host.docker.internal' ? 'localhost' : robot.task_server_info.host
      const robotUrl = `http://${host}:${robot.task_server_info.port}/do_task`

      const res = await fetch(robotUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          task_description: taskDescription
        }),
      })

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }

      const result = await res.json()

      setResponse({
        success: result.success,
        message: result.message,
        replan: result.replan,
        timestamp: new Date()
      })

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send task')
    } finally {
      setIsSending(false)
    }
  }

  const isReachable = health?.reachable === true

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_0.8fr] gap-6">
      {/* Left column: task form + response */}
      <div className="space-y-4">
        {/* Connection Status */}
        <div className="flex items-center gap-2 px-1">
          {isReachable ? (
            <CheckCircle className="w-4 h-4 text-emerald-400" />
          ) : (
            <XCircle className="w-4 h-4 text-red-400" />
          )}
          <span className={cn(
            'text-sm font-medium',
            isReachable ? 'text-emerald-400' : 'text-red-400'
          )}>
            {isReachable ? 'Connected' : 'Disconnected'}
          </span>
          <span className="text-xs text-[var(--color-text-muted)] font-mono ml-auto">
            {robot.task_server_info?.host}:{robot.task_server_info?.port}
          </span>
        </div>

        {/* Task Input */}
        <div>
          <textarea
            value={taskDescription}
            onChange={(e) => setTaskDescription(e.target.value)}
            placeholder="Enter a natural language task description (e.g., 'navigate to the kitchen and pick up the red cup')"
            className="w-full h-28 px-3 py-2 bg-surface-overlay border border-border rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 resize-none text-sm"
            disabled={!isReachable || isSending}
          />

          {error && (
            <div className="mt-2 p-2.5 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          <div className="flex justify-end mt-3">
            <Button
              onClick={handleSendTask}
              disabled={!isReachable || isSending || !taskDescription.trim()}
              className="flex items-center gap-2"
            >
              {isSending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              {isSending ? 'Executing...' : 'Send Task'}
            </Button>
          </div>
        </div>

        {/* Execution Status */}
        {isSending && (
          <div className="p-4 rounded-lg border-2 border-amber-500/40 bg-amber-500/5">
            <div className="flex items-center gap-2 mb-2">
              <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />
              <span className="text-sm font-medium text-amber-300">Executing Task</span>
            </div>
            <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{taskDescription}</p>
          </div>
        )}

        {/* Response Display */}
        {response && (
          <div className={cn(
            'p-4 rounded-lg border',
            response.success
              ? 'border-emerald-500/30 bg-emerald-500/5'
              : 'border-red-500/30 bg-red-500/5'
          )}>
            <div className="flex items-center gap-2 mb-2">
              {response.success ? (
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              ) : (
                <XCircle className="w-4 h-4 text-red-400" />
              )}
              <span className={cn(
                'text-sm font-medium',
                response.success ? 'text-emerald-300' : 'text-red-300'
              )}>
                {response.success ? 'Task Completed' : 'Task Failed'}
              </span>
              <span className="text-xs text-[var(--color-text-muted)] ml-auto">
                {response.timestamp.toLocaleTimeString()}
              </span>
            </div>
            <pre className={cn(
              'whitespace-pre-wrap font-mono text-xs leading-relaxed mt-2',
              response.success ? 'text-emerald-300/80' : 'text-red-300/80'
            )}>
              {response.message}
            </pre>
            {response.replan && (
              <div className="mt-3 flex items-center gap-2">
                <RotateCcw className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-xs font-medium text-amber-300">Replan required</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right column: telemetry panels */}
      <div className="space-y-4">
        {/* Video feed */}
        <div className="rounded-lg border border-border bg-surface-overlay/60 overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border-subtle">
            <Video className="w-4 h-4 text-[var(--color-text-muted)]" />
            <span className="text-sm font-medium text-[var(--color-text-secondary)]">Live Video Feed</span>
          </div>
          <div className="aspect-video flex flex-col items-center justify-center text-[var(--color-text-muted)] bg-surface/50">
            <Video className="w-10 h-10 mb-3 opacity-40" />
            <p className="text-sm">Video feed not connected</p>
            <p className="text-xs mt-1 font-mono opacity-50">
              /robots/{robot.robot_id}/video
            </p>
          </div>
        </div>

        {/* Joint states */}
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
              /robots/{robot.robot_id}/joints
            </p>
          </div>
        </div>

        {/* Command log */}
        <div className="rounded-lg border border-border bg-surface-overlay/60 overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border-subtle">
            <Terminal className="w-4 h-4 text-[var(--color-text-muted)]" />
            <span className="text-sm font-medium text-[var(--color-text-secondary)]">Joint Commands</span>
          </div>
          <div className="h-32 flex flex-col items-center justify-center text-[var(--color-text-muted)] bg-surface/50">
            <Terminal className="w-8 h-8 mb-2 opacity-40" />
            <p className="text-xs">Command stream not connected</p>
            <p className="text-xs mt-1 font-mono opacity-50">
              /robots/{robot.robot_id}/commands
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Register Robot Modal (Simplified)
// =============================================================================

function RegisterRobotModal({ 
  isOpen, 
  onClose 
}: { 
  isOpen: boolean
  onClose: () => void
}) {
  const [configPath, setConfigPath] = useState('')
  const [robotId, setRobotId] = useState('')
  const [host, setHost] = useState('localhost')
  const [port, setPort] = useState('')
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: robotsApi.register,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['robots'] })
      queryClient.invalidateQueries({ queryKey: ['robotsHealth'] })
      onClose()
      setConfigPath('')
      setRobotId('')
      setHost('localhost')
      setPort('')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({
      config_path: configPath,
      robot_id: robotId,
      host: host,
      port: parseInt(port, 10),
    })
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Register Robot">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-[var(--color-text)] mb-2">
            YAML Config Path
          </label>
          <input
            type="text"
            value={configPath}
            onChange={(e) => setConfigPath(e.target.value)}
            placeholder="robot_fleet/robots/examples/moma/moma.yaml"
            className="w-full px-4 py-2 bg-surface-overlay border border-border rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 font-mono text-sm"
            required
          />
          <p className="text-xs text-[var(--color-text-muted)] mt-1">
            Path to the embodiment YAML (relative to project root)
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-[var(--color-text)] mb-2">
            Robot ID
          </label>
          <input
            type="text"
            value={robotId}
            onChange={(e) => setRobotId(e.target.value)}
            placeholder="moma-kitchen"
            className="w-full px-4 py-2 bg-surface-overlay border border-border rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 font-mono text-sm"
            required
          />
          <p className="text-xs text-[var(--color-text-muted)] mt-1">
            Unique identifier for this robot instance
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text)] mb-2">
              Host / IP
            </label>
            <input
              type="text"
              value={host}
              onChange={(e) => setHost(e.target.value)}
              placeholder="localhost"
              className="w-full px-4 py-2 bg-surface-overlay border border-border rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 font-mono text-sm"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-[var(--color-text)] mb-2">
              Port
            </label>
            <input
              type="number"
              value={port}
              onChange={(e) => setPort(e.target.value)}
              placeholder="8001"
              min={1}
              max={65535}
              className="w-full px-4 py-2 bg-surface-overlay border border-border rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 font-mono text-sm"
              required
            />
          </div>
        </div>
        <p className="text-xs text-[var(--color-text-muted)] -mt-2">
          For Docker: localhost + exposed port. For real robots: IP + server port.
        </p>

        {mutation.error && (
          <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-sm text-red-400">{(mutation.error as Error).message}</p>
          </div>
        )}

        <div className="flex justify-end gap-3 pt-4">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? 'Registering...' : 'Register Robot'}
          </Button>
        </div>
      </form>
    </Modal>
  )
}

// =============================================================================
// Main Robots Page
// =============================================================================

export function Robots() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false)
  const [selectedRobot, setSelectedRobot] = useState<Robot | null>(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [checkingRobots, setCheckingRobots] = useState<Set<string>>(new Set())
  const [isCheckingAll, setIsCheckingAll] = useState(false)
  const [isRefreshingAll, setIsRefreshingAll] = useState(false)
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null)
  const queryClient = useQueryClient()

  // Enable real-time updates for robot allocations and health
  useRealtimeUpdates()

  const { data: robots = [], isLoading, error } = useQuery({
    queryKey: ['robots'],
    queryFn: () => robotsApi.list()
  })

  // Robot health: event-driven only. Fetched on mount and when WebSocket sends invalidate for 'robot-health'
  // (triggered by telemetry.health_changed from Telemetry when heartbeat changes effective reachable status).
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
    // No refetchInterval — updates only via WebSocket invalidation from Telemetry events
    refetchOnWindowFocus: false,
    refetchOnReconnect: true,
  })

  // Restore modal state from URL params
  useEffect(() => {
    const robotParam = searchParams.get('robot')
    const tabParam = searchParams.get('tab')

    if (robotParam && robots.length > 0) {
      const robot = robots.find(r => r.robot_id === robotParam)
      if (robot) {
        setSelectedRobot(robot)
        setActiveTab(tabParam || 'overview')
      }
    }
  }, [robots, searchParams])

  // Update URL when modal state changes (but don't clear params that should restore modal)
  useEffect(() => {
    const currentRobotParam = searchParams.get('robot')
    const currentTabParam = searchParams.get('tab')

    if (selectedRobot) {
      // Update URL to match current modal state
      if (currentRobotParam !== selectedRobot.robot_id || currentTabParam !== activeTab) {
        setSearchParams({
          robot: selectedRobot.robot_id,
          tab: activeTab
        })
      }
    }
    // Don't clear URL params here - let them persist for direct links
  }, [selectedRobot, activeTab, searchParams, setSearchParams])

  // Fetch allocations for all robots (don't let this block robots display)
  const { data: robotAllocations = {} } = useQuery({
    queryKey: ['robot-allocations'],
    queryFn: async () => {
      if (robots.length === 0) return {}
      const allocations: Record<string, RobotAllocations> = {}
      for (const robot of robots) {
        try {
          const alloc = await robotsApi.getAllocations(robot.robot_id)
          allocations[robot.robot_id] = alloc
        } catch (error) {
          console.error(`Failed to fetch allocations for ${robot.robot_id}:`, error)
          // If allocation fetch fails, provide default empty allocation
          allocations[robot.robot_id] = {
            robot_id: robot.robot_id,
            plans_count: 0,
            goals_count: 0,
            tasks_count: 0,
            plans: [],
            goals: []
          }
        }
      }
      return allocations
    },
    enabled: robots.length > 0,
    // Don't retry on failure to prevent blocking
    retry: false,
    // Don't refetch on window focus to avoid spam
    refetchOnWindowFocus: false,
  })

  // Debug logging (after both queries are declared)
  console.log('Robots data:', { robots, isLoading, error, robotAllocations })

  const deleteMutation = useMutation({
    mutationFn: robotsApi.unregister,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['robots'] })
    },
  })

  // Check health of a single robot
  const checkSingleHealth = async (robotId: string) => {
    setCheckingRobots(prev => new Set(prev).add(robotId))
    try {
      // Optimistic update: assume robot is reachable while checking
      queryClient.setQueryData(['robot-health'], (oldData: any) => ({
        ...oldData,
        [robotId]: { ...oldData?.[robotId], reachable: true, checking: true }
      }))

      // Trigger immediate refetch
      await queryClient.invalidateQueries({ queryKey: ['robot-health'] })
    } catch (error) {
      console.error('Failed to refresh health:', error)
    } finally {
      setCheckingRobots(prev => {
        const newSet = new Set(prev)
        newSet.delete(robotId)
        return newSet
      })
    }
  }

  const checkAllHealth = async () => {
    setIsCheckingAll(true)
    try {
      // Trigger immediate refetch for all robots
      await queryClient.invalidateQueries({ queryKey: ['robot-health'] })
    } catch (error) {
      console.error('Failed to refresh health:', error)
    } finally {
      setIsCheckingAll(false)
    }
  }


  // Refresh all robots from YAML
  const refreshAllFromYaml = async () => {
    setIsRefreshingAll(true)
    setRefreshMessage(null)
    try {
      const response = await fetch('/api/robots/refresh/all', { method: 'POST' })
      const data = await response.json()
      
      if (response.ok) {
        queryClient.invalidateQueries({ queryKey: ['robots'] })
        setRefreshMessage(`✓ Refreshed ${data.success_count}/${data.total} robots from YAML`)
        // Also refresh health after YAML refresh
        setTimeout(checkAllHealth, 500)
      } else {
        setRefreshMessage(`✗ ${data.detail || 'Failed to refresh'}`)
      }
    } catch (error) {
      setRefreshMessage('✗ Failed to refresh robots from YAML')
    }
    setIsRefreshingAll(false)
    // Clear message after 5 seconds
    setTimeout(() => setRefreshMessage(null), 5000)
  }

  if (isLoading) {
    return <div className="text-[var(--color-text-secondary)]">Loading...</div>
  }

  const connectedCount = Object.values(robotHealth).filter(h => h.reachable).length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Robot Fleet</h1>
          <p className="text-[var(--color-text-secondary)]">
            {robots.length} registered • {connectedCount} connected
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button 
            variant="secondary" 
            onClick={refreshAllFromYaml}
            disabled={isRefreshingAll || robots.length === 0}
            title="Re-read all YAML files and update robot capabilities"
          >
            <Download className={cn("w-4 h-4", isRefreshingAll && "animate-spin")} />
            {isRefreshingAll ? 'Syncing...' : 'Sync YAMLs'}
          </Button>
          <Button 
            variant="secondary" 
            onClick={checkAllHealth}
            disabled={isCheckingAll || robots.length === 0}
          >
            <RefreshCw className={cn("w-4 h-4", isCheckingAll && "animate-spin")} />
            {isCheckingAll ? 'Checking...' : 'Check All'}
          </Button>
          <Button onClick={() => setIsRegisterModalOpen(true)}>
            <Plus className="w-4 h-4" />
            Register Robot
          </Button>
        </div>
      </div>

      {/* Refresh Message */}
      {refreshMessage && (
        <div className={cn(
          'p-3 rounded-lg border text-sm',
          refreshMessage.startsWith('✓') 
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            : 'bg-red-500/10 border-red-500/20 text-red-400'
        )}>
          {refreshMessage}
        </div>
      )}

      {/* Robots Grid */}
      {robots.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {robots.map((robot) => (
            <RobotCard
              key={robot.robot_id}
              robot={robot}
              health={robotHealth[robot.robot_id]}
              allocations={robotAllocations[robot.robot_id]}
              onClick={() => setSelectedRobot(robot)}
              onDelete={(id) => deleteMutation.mutate(id)}
              onCheckHealth={checkSingleHealth}
              isChecking={checkingRobots.has(robot.robot_id)}
            />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<Bot className="w-8 h-8" />}
          title="No robots registered"
          description="Register your first robot to start building your fleet. You'll need the YAML config path, host/IP, and port."
          action={
            <Button onClick={() => setIsRegisterModalOpen(true)}>
              <Plus className="w-4 h-4" />
              Register Robot
            </Button>
          }
        />
      )}

      {/* Modals */}
      <RegisterRobotModal 
        isOpen={isRegisterModalOpen} 
        onClose={() => setIsRegisterModalOpen(false)} 
      />
      
      <RobotDetailModal
        robot={selectedRobot}
        health={selectedRobot ? robotHealth[selectedRobot.robot_id] : undefined}
        isOpen={!!selectedRobot}
        onClose={() => {
          setSelectedRobot(null)
          setActiveTab('overview')
          // Clear URL params when modal is closed
          setSearchParams({})
        }}
        onRefresh={checkAllHealth}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
    </div>
  )
}
