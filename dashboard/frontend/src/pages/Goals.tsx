import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Target, Plus, Trash2, GitBranch, Users } from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { goalsApi, plansApi } from '../lib/api'

function PlansModal({ goalId, isOpen, onClose, plans, goals }: { goalId: number | null; isOpen: boolean; onClose: () => void; plans: any[]; goals: any[] }) {
  const goalPlans = plans.filter(plan => plan.goal_ids.includes(goalId))
  const goal = goals.find(g => g.goal_id === goalId)
  const goalIdStr = goalId?.toString()

  // Fetch full plan details for each plan to get task data
  const planDetailsQueries = useQuery({
    queryKey: ['plans-details', goalPlans.map(p => p.plan_id)],
    queryFn: async () => {
      const details = await Promise.all(
        goalPlans.map(plan => plansApi.get(plan.plan_id))
      )
      return details
    },
    enabled: isOpen && goalPlans.length > 0,
  })

  const fullPlans = planDetailsQueries.data || []

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Plans for Goal #${goalId}`}>
      <div className="space-y-4">
        {goal && (
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-600/50">
            <h4 className="font-medium text-white mb-2">Goal Description</h4>
            <p className="text-slate-300">{goal.description}</p>
          </div>
        )}

        {planDetailsQueries.isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-slate-400">Loading plan details...</div>
          </div>
        ) : goalPlans.length > 0 ? (
          <div className="space-y-3">
            <h4 className="font-medium text-white">Generated Plans ({goalPlans.length})</h4>
            {goalPlans.map(plan => {
              const fullPlan = fullPlans.find(fp => fp.plan_id === plan.plan_id)
              // Use task_descriptions from allocation_artifacts instead of tasks array
              const taskDescriptions = fullPlan?.allocation_artifacts?.task_descriptions || []
              const goalTasks = taskDescriptions.filter((task: any) => task.goal_id === goalIdStr) || []

              // Debug logging
              console.log('Plan:', plan.plan_id, 'FullPlan:', fullPlan, 'TaskDescriptions:', taskDescriptions, 'GoalTasks:', goalTasks, 'GoalIdStr:', goalIdStr)
              return (
                <Card key={plan.plan_id} className="p-4 bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
                  <div className="flex items-center justify-between">
                    <div>
                      <h5 className="font-medium text-white">Plan #{plan.plan_id}</h5>
                      <div className="text-sm text-slate-400 mt-1">
                        {goalTasks.length} tasks solving this goal • {taskDescriptions.length || plan.task_ids.length} total tasks
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded text-xs ${
                        plan.planning_strategy === 'manual'
                          ? 'bg-amber-500/20 border border-amber-500/40 text-amber-300'
                          : 'bg-violet-500/20 border border-violet-500/40 text-violet-300'
                      }`}>
                        {plan.planning_strategy}
                      </span>
                      <span className={`px-2 py-1 rounded text-xs ${
                        plan.allocation_strategy === 'none'
                          ? 'bg-slate-500/20 border border-slate-500/40 text-slate-300'
                          : 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-300'
                      }`}>
                        {plan.allocation_strategy}
                      </span>
                    </div>
                  </div>
                </Card>
              )
            })}
          </div>
        ) : (
          <div className="text-center py-8">
            <GitBranch className="w-12 h-12 text-slate-500 mx-auto mb-4" />
            <p className="text-slate-400">No plans have been generated to solve this goal yet.</p>
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
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Goal Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe what you want the robots to accomplish..."
            rows={4}
            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 resize-none"
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
  const queryClient = useQueryClient()

  const { data: goals = [], isLoading } = useQuery({
    queryKey: ['goals'],
    queryFn: goalsApi.list,
  })

  const { data: plans = [] } = useQuery({
    queryKey: ['plans'],
    queryFn: plansApi.list,
  })

  // Calculate plans count for each goal
  const getPlansCountForGoal = (goalId: number) => {
    return plans.filter(plan => plan.goal_ids.includes(goalId)).length
  }

  const deleteMutation = useMutation({
    mutationFn: goalsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['goals'] })
    },
  })

  if (isLoading) {
    return <div className="text-slate-400">Loading...</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Goals</h1>
          <p className="text-slate-400">{goals.length} goals created</p>
        </div>
        <Button onClick={() => setIsModalOpen(true)}>
          <Plus className="w-4 h-4" />
          Create Goal
        </Button>
      </div>

      {/* Goals List */}
      {goals.length > 0 ? (
        <div className="space-y-3">
          {goals.map((goal) => (
            <Card key={goal.goal_id} className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center flex-shrink-0">
                <Target className="w-5 h-5 text-violet-400" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="font-medium text-white">{goal.description}</p>
                    <p className="text-xs text-slate-500 mt-1 font-mono">
                      Goal #{goal.goal_id} • {goal.task_ids.length} tasks • {getPlansCountForGoal(goal.goal_id)} plans
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setPlansModalGoal(goal.goal_id)}
                      className="text-blue-400 hover:text-blue-300 hover:bg-blue-500/10"
                    >
                      <GitBranch className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deleteMutation.mutate(goal.goal_id)}
                      className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : (
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
      )}

      <CreateGoalModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
      <PlansModal
        goalId={plansModalGoal}
        isOpen={plansModalGoal !== null}
        onClose={() => setPlansModalGoal(null)}
        plans={plans}
        goals={goals}
      />
    </div>
  )
}
