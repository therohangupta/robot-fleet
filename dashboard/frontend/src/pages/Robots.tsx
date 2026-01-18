import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Bot, Plus, Trash2, Wifi, WifiOff, RefreshCw, CheckCircle, XCircle, Loader2, FileCode, Download } from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { robotsApi } from '../lib/api'
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
// Robot Card with Health Status
// =============================================================================

function RobotCard({ 
  robot, 
  health,
  onClick,
  onDelete,
  onCheckHealth
}: { 
  robot: Robot
  health?: RobotHealth
  onClick: () => void
  onDelete: (id: string) => void
  onCheckHealth: (id: string) => void
}) {
  const isReachable = health?.reachable ?? false
  const isChecking = health === undefined
  
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
              isReachable ? 'bg-emerald-500/10' : 'bg-slate-800'
            )}>
              <Bot className={cn('w-5 h-5', isReachable ? 'text-emerald-400' : 'text-slate-500')} />
            </div>
            <div>
              <h3 className="font-semibold text-white">{robot.robot_id}</h3>
              <p className="text-xs text-slate-500">{robot.robot_type}</p>
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

        {/* Capabilities */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {robot.capabilities.slice(0, 3).map((cap, i) => (
            <span 
              key={i}
              className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-400 font-mono"
            >
              {cap.length > 15 ? cap.substring(0, 15) + '...' : cap}
            </span>
          ))}
          {robot.capabilities.length > 3 && (
            <span className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-500">
              +{robot.capabilities.length - 3}
            </span>
          )}
        </div>

        {/* Connection Info & Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-slate-800">
          <div className="flex items-center gap-2 text-xs text-slate-500">
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
              className="text-slate-400 hover:text-white"
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

// =============================================================================
// Robot Detail Modal
// =============================================================================

function RobotDetailModal({ 
  robot,
  health,
  isOpen, 
  onClose,
  onRefresh
}: { 
  robot: Robot | null
  health?: RobotHealth
  isOpen: boolean
  onClose: () => void
  onRefresh: () => void
}) {
  const [yamlDetails, setYamlDetails] = useState<RobotYamlDetails | null>(null)
  const [isLoadingYaml, setIsLoadingYaml] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null)
  const queryClient = useQueryClient()

  // Fetch YAML details when modal opens
  useEffect(() => {
    if (isOpen && robot) {
      setIsLoadingYaml(true)
      fetch(`/api/robots/${robot.robot_id}/yaml`)
        .then(res => res.json())
        .then(data => {
          setYamlDetails(data)
          setIsLoadingYaml(false)
        })
        .catch(() => setIsLoadingYaml(false))
    }
  }, [isOpen, robot])

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
        onRefresh()
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

  if (!robot) return null

  const isReachable = health?.reachable ?? false

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Robot: ${robot.robot_id}`}>
      <div className="space-y-6">
        {/* Status Banner */}
        <div className={cn(
          'p-4 rounded-lg border',
          isReachable 
            ? 'bg-emerald-500/10 border-emerald-500/20' 
            : 'bg-red-500/10 border-red-500/20'
        )}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {isReachable ? (
                <CheckCircle className="w-5 h-5 text-emerald-400" />
              ) : (
                <XCircle className="w-5 h-5 text-red-400" />
              )}
              <div>
                <p className={cn('font-medium', isReachable ? 'text-emerald-400' : 'text-red-400')}>
                  {isReachable ? 'Connected' : 'Unreachable'}
                </p>
                <p className="text-xs text-slate-400">
                  {robot.task_server_info?.host}:{robot.task_server_info?.port}
                  {health?.latency_ms && ` • ${Math.round(health.latency_ms)}ms`}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Basic Info */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-slate-500 mb-1">Robot Type</p>
            <p className="text-white font-mono">{robot.robot_type}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500 mb-1">Description</p>
            <p className="text-white text-sm">{robot.description || 'No description'}</p>
          </div>
        </div>

        {/* Capabilities */}
        <div>
          <p className="text-xs text-slate-500 mb-2">Capabilities ({robot.capabilities.length})</p>
          <div className="flex flex-wrap gap-2">
            {robot.capabilities.map((cap, i) => (
              <span 
                key={i}
                className="px-3 py-1.5 bg-slate-800 rounded-lg text-sm text-slate-300"
              >
                {cap}
              </span>
            ))}
          </div>
        </div>

        {/* YAML Source */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <FileCode className="w-4 h-4 text-slate-400" />
              <p className="text-xs text-slate-500">YAML Source</p>
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
            <div className="p-4 bg-slate-800/50 rounded-lg text-slate-400 text-sm">
              Loading YAML details...
            </div>
          ) : yamlDetails?.yaml_path ? (
            <div className="space-y-2">
              <p className="text-xs text-slate-400 font-mono">{yamlDetails.yaml_path}</p>
              <pre className="p-4 bg-slate-900 rounded-lg text-xs text-slate-300 overflow-auto max-h-48 font-mono">
                {JSON.stringify(yamlDetails.yaml_content, null, 2)}
              </pre>
            </div>
          ) : (
            <div className="p-4 bg-slate-800/50 rounded-lg text-slate-500 text-sm">
              YAML source file not found
            </div>
          )}
          
          {refreshMessage && (
            <p className={cn(
              'text-sm mt-2',
              refreshMessage.startsWith('✓') ? 'text-emerald-400' : 'text-red-400'
            )}>
              {refreshMessage}
            </p>
          )}
        </div>

        {/* Close Button */}
        <div className="flex justify-end pt-4 border-t border-slate-800">
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </Modal>
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
          <label className="block text-sm font-medium text-slate-300 mb-2">
            YAML Config Path
          </label>
          <input
            type="text"
            value={configPath}
            onChange={(e) => setConfigPath(e.target.value)}
            placeholder="robot_fleet/robots/examples/moma/moma.yaml"
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 font-mono text-sm"
            required
          />
          <p className="text-xs text-slate-500 mt-1">
            Path to the embodiment YAML (relative to project root)
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Robot ID
          </label>
          <input
            type="text"
            value={robotId}
            onChange={(e) => setRobotId(e.target.value)}
            placeholder="moma-kitchen"
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 font-mono text-sm"
            required
          />
          <p className="text-xs text-slate-500 mt-1">
            Unique identifier for this robot instance
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Host / IP
            </label>
            <input
              type="text"
              value={host}
              onChange={(e) => setHost(e.target.value)}
              placeholder="localhost"
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 font-mono text-sm"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Port
            </label>
            <input
              type="number"
              value={port}
              onChange={(e) => setPort(e.target.value)}
              placeholder="8001"
              min={1}
              max={65535}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 font-mono text-sm"
              required
            />
          </div>
        </div>
        <p className="text-xs text-slate-500 -mt-2">
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
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false)
  const [selectedRobot, setSelectedRobot] = useState<Robot | null>(null)
  const [healthStatus, setHealthStatus] = useState<Record<string, RobotHealth>>({})
  const [isCheckingAll, setIsCheckingAll] = useState(false)
  const [isRefreshingAll, setIsRefreshingAll] = useState(false)
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const { data: robots = [], isLoading } = useQuery({
    queryKey: ['robots'],
    queryFn: () => robotsApi.list(),
  })

  const deleteMutation = useMutation({
    mutationFn: robotsApi.unregister,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['robots'] })
    },
  })

  // Check health of a single robot
  const checkSingleHealth = async (robotId: string) => {
    try {
      const response = await fetch(`/api/robots/${robotId}/health`)
      const health = await response.json()
      setHealthStatus(prev => ({ ...prev, [robotId]: health }))
    } catch (error) {
      setHealthStatus(prev => ({ 
        ...prev, 
        [robotId]: { robot_id: robotId, host: '', port: 0, reachable: false, error: 'Failed to check' }
      }))
    }
  }

  // Check health of all robots
  const checkAllHealth = async () => {
    setIsCheckingAll(true)
    try {
      const response = await fetch('/api/robots/health/all')
      const data = await response.json()
      const newStatus: Record<string, RobotHealth> = {}
      for (const health of data.robots) {
        newStatus[health.robot_id] = health
      }
      setHealthStatus(newStatus)
    } catch (error) {
      console.error('Failed to check health:', error)
    }
    setIsCheckingAll(false)
  }

  // Check health on initial load and when robots change
  useEffect(() => {
    if (robots.length > 0) {
      checkAllHealth()
    }
  }, [robots.length])

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
    return <div className="text-slate-400">Loading...</div>
  }

  const connectedCount = Object.values(healthStatus).filter(h => h.reachable).length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Robot Fleet</h1>
          <p className="text-slate-400">
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
              health={healthStatus[robot.robot_id]}
              onClick={() => setSelectedRobot(robot)}
              onDelete={(id) => deleteMutation.mutate(id)}
              onCheckHealth={checkSingleHealth}
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
        health={selectedRobot ? healthStatus[selectedRobot.robot_id] : undefined}
        isOpen={!!selectedRobot}
        onClose={() => setSelectedRobot(null)}
        onRefresh={checkAllHealth}
      />
    </div>
  )
}
