import { useQuery } from '@tanstack/react-query'
import { Bot, Target, GitBranch, CheckCircle, Clock, AlertCircle } from 'lucide-react'
import { Card } from '../components/common/Card'
import { StatusBadge } from '../components/common/StatusBadge'
import { robotsApi, goalsApi, plansApi, tasksApi } from '../lib/api'
import { cn } from '../lib/utils'
import type { Task } from '../types'

function StatCard({ 
  icon: Icon, 
  label, 
  value, 
  subValue,
  color 
}: { 
  icon: typeof Bot
  label: string
  value: number | string
  subValue?: string
  color: string
}) {
  return (
    <Card hover className="relative overflow-hidden">
      <div className={cn('absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl opacity-10', color)} />
      <div className="relative">
        <div className={cn('w-10 h-10 rounded-lg flex items-center justify-center mb-3', color.replace('bg-', 'bg-').concat('/10'))}>
          <Icon className={cn('w-5 h-5', color.replace('bg-', 'text-'))} />
        </div>
        <p className="text-3xl font-bold text-white mb-1">{value}</p>
        <p className="text-sm text-slate-400">{label}</p>
        {subValue && <p className="text-xs text-slate-500 mt-1">{subValue}</p>}
      </div>
    </Card>
  )
}

function RecentActivity({ tasks }: { tasks: Task[] }) {
  const recentTasks = tasks.slice(0, 8)
  
  return (
    <Card>
      <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
      <div className="space-y-3">
        {recentTasks.map((task) => (
          <div key={task.task_id} className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/30">
            <div className={cn(
              'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
              task.status === 'completed' ? 'bg-emerald-500/10 text-emerald-400' :
              task.status === 'in_progress' ? 'bg-amber-500/10 text-amber-400' :
              task.status === 'failed' ? 'bg-red-500/10 text-red-400' :
              'bg-slate-700 text-slate-400'
            )}>
              {task.status === 'completed' ? <CheckCircle className="w-4 h-4" /> :
               task.status === 'in_progress' ? <Clock className="w-4 h-4 animate-pulse" /> :
               task.status === 'failed' ? <AlertCircle className="w-4 h-4" /> :
               <Clock className="w-4 h-4" />}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-white truncate font-mono">{task.description}</p>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-slate-500">Task #{task.task_id}</span>
                {task.robot_id && (
                  <span className="text-xs text-cyber-400">@{task.robot_id}</span>
                )}
              </div>
            </div>
            <StatusBadge status={task.status} />
          </div>
        ))}
        {recentTasks.length === 0 && (
          <p className="text-sm text-slate-500 text-center py-8">No recent activity</p>
        )}
      </div>
    </Card>
  )
}

export function Dashboard() {
  const { data: robots = [] } = useQuery({
    queryKey: ['robots'],
    queryFn: () => robotsApi.list(),
  })

  // Fetch robot health statuses for accurate online status
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
    refetchInterval: 5000, // Check health every 5 seconds
  })

  const { data: goals = [] } = useQuery({
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

  // Count robots that are both registered/running AND currently reachable
  const onlineRobots = robots.filter(r => {
    const isStatusOk = r.status && r.status !== 'unknown' && r.status !== 'error' && r.status !== 'stopped'
    const isReachable = robotHealth ? robotHealth[r.robot_id]?.reachable === true : true // Assume online if no health data yet
    return isStatusOk && isReachable
  })
  const completedTasks = tasks.filter(t => t.status === 'completed')
  const inProgressTasks = tasks.filter(t => t.status === 'in_progress')

  return (
    <div className="space-y-6">
      {/* Welcome */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-2">
          Welcome to <span className="gradient-text">Mission Control</span>
        </h1>
        <p className="text-slate-400">
          Monitor and manage your robot fleet from one central dashboard.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Bot}
          label="Active Robots"
          value={onlineRobots.length}
          subValue={`${robots.length} total registered`}
          color="bg-cyber-500"
        />
        <StatCard
          icon={Target}
          label="Goals"
          value={goals.length}
          subValue="Pending completion"
          color="bg-violet-500"
        />
        <StatCard
          icon={GitBranch}
          label="Plans"
          value={plans.length}
          subValue={`${inProgressTasks.length} tasks running`}
          color="bg-amber-500"
        />
        <StatCard
          icon={CheckCircle}
          label="Completed Tasks"
          value={completedTasks.length}
          subValue={`${tasks.length} total tasks`}
          color="bg-emerald-500"
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Feed */}
        <div className="lg:col-span-2">
          <RecentActivity tasks={tasks} />
        </div>

        {/* Robot Status */}
        <div>
          <Card>
            <h3 className="text-lg font-semibold text-white mb-4">Robot Status</h3>
            <div className="space-y-3">
              {robots.map((robot) => (
                <div key={robot.robot_id} className="flex items-center justify-between p-3 rounded-lg bg-slate-800/30">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      'w-2 h-2 rounded-full',
                      (robot.status === 'running' || robot.status === 'registered')
                        ? 'bg-emerald-400 animate-pulse'
                        : 'bg-slate-500'
                    )} />
                    <div>
                      <p className="text-sm font-medium text-white">{robot.robot_id}</p>
                      <p className="text-xs text-slate-500">{robot.robot_type}</p>
                    </div>
                  </div>
                  <span className="text-xs text-slate-400 font-mono">
                    {robot.task_server_info?.port}
                  </span>
                </div>
              ))}
              {robots.length === 0 && (
                <p className="text-sm text-slate-500 text-center py-4">No robots registered</p>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
