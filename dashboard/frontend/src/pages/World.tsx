import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Globe, Plus, Trash2, MessageSquare } from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { worldApi } from '../lib/api'
import { formatDate } from '../lib/utils'

function AddStatementModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [statement, setStatement] = useState('')
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: worldApi.add,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['world'] })
      onClose()
      setStatement('')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({ statement })
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Add World Statement">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Statement
          </label>
          <textarea
            value={statement}
            onChange={(e) => setStatement(e.target.value)}
            placeholder="Describe a fact about the world state..."
            rows={4}
            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500 resize-none"
            required
          />
          <p className="text-xs text-slate-500 mt-2">
            Examples: "There is a kitchen and a living room", "The kitchen has 2 cups", "The robot is at the entrance"
          </p>
        </div>

        {mutation.error && (
          <p className="text-sm text-red-400">{(mutation.error as Error).message}</p>
        )}

        <div className="flex justify-end gap-3 pt-4">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? 'Adding...' : 'Add Statement'}
          </Button>
        </div>
      </form>
    </Modal>
  )
}

export function World() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const queryClient = useQueryClient()

  const { data: statements = [], isLoading } = useQuery({
    queryKey: ['world'],
    queryFn: worldApi.list,
  })

  const deleteMutation = useMutation({
    mutationFn: worldApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['world'] })
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
          <h1 className="text-2xl font-bold text-white">World State</h1>
          <p className="text-slate-400">{statements.length} statements defined</p>
        </div>
        <Button onClick={() => setIsModalOpen(true)}>
          <Plus className="w-4 h-4" />
          Add Statement
        </Button>
      </div>

      {/* Info Card */}
      <Card className="border-cyber-500/30 bg-cyber-500/5">
        <div className="flex items-start gap-3">
          <Globe className="w-5 h-5 text-cyber-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-medium text-white mb-1">About World State</h3>
            <p className="text-sm text-slate-400">
              World statements describe the current state of the environment. The planner uses these 
              statements to understand the context and generate appropriate task plans. Be descriptive 
              about locations, objects, and their relationships.
            </p>
          </div>
        </div>
      </Card>

      {/* Statements List */}
      {statements.length > 0 ? (
        <div className="space-y-3">
          {statements.map((ws) => (
            <Card key={ws.id} className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center flex-shrink-0">
                <MessageSquare className="w-5 h-5 text-emerald-400" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-white">{ws.statement}</p>
                    <p className="text-xs text-slate-500 mt-1 font-mono">
                      ID: {ws.id.substring(0, 8)}... • {formatDate(ws.created_at)}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMutation.mutate(ws.id)}
                    className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<Globe className="w-8 h-8" />}
          title="No world statements"
          description="Add statements to describe the current state of your environment for the planner."
          action={
            <Button onClick={() => setIsModalOpen(true)}>
              <Plus className="w-4 h-4" />
              Add Statement
            </Button>
          }
        />
      )}

      <AddStatementModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </div>
  )
}
