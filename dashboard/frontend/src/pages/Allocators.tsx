import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { FileText, ChevronRight, Variable } from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { methodsApi, type MethodSummary } from '../lib/api'
import { cn } from '../lib/utils'

// =============================================================================
// Prompt Card Component
// =============================================================================

// Color palette for automatic assignment
const COLOR_PALETTE = [
  { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', icon: 'bg-blue-500/20' },
  { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', icon: 'bg-emerald-500/20' },
  { bg: 'bg-violet-500/10', border: 'border-violet-500/30', text: 'text-violet-400', icon: 'bg-violet-500/20' },
  { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', icon: 'bg-amber-500/20' },
  { bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-400', icon: 'bg-rose-500/20' },
  { bg: 'bg-cyan-500/10', border: 'border-cyan-500/30', text: 'text-cyan-400', icon: 'bg-cyan-500/20' },
  { bg: 'bg-pink-500/10', border: 'border-pink-500/30', text: 'text-pink-400', icon: 'bg-pink-500/20' },
  { bg: 'bg-teal-500/10', border: 'border-teal-500/30', text: 'text-teal-400', icon: 'bg-teal-500/20' },
  { bg: 'bg-indigo-500/10', border: 'border-indigo-500/30', text: 'text-indigo-400', icon: 'bg-indigo-500/20' },
  { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', icon: 'bg-orange-500/20' },
]

// Generate colors for a planner type using a simple hash
function getPlannerColors(plannerType: string) {
  // Simple hash function for consistent color assignment
  let hash = 0
  for (let i = 0; i < plannerType.length; i++) {
    hash = ((hash << 5) - hash) + plannerType.charCodeAt(i)
    hash = hash & hash // Convert to 32-bit integer
  }
  return COLOR_PALETTE[Math.abs(hash) % COLOR_PALETTE.length]
}

function MethodCard({
  method,
  onClick
}: {
  method: MethodSummary
  onClick: () => void
}) {
  const colors = getPlannerColors(method.type)

  return (
    <Card
      className={cn('cursor-pointer hover:border-slate-600 transition-all group', colors.bg, colors.border)}
      onClick={onClick}
    >
      <div className="flex items-start gap-4">
        <div className={cn('w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0', colors.icon)}>
          <FileText className={cn('w-6 h-6', colors.text)} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-white">{method.name}</h3>
            <span className={cn('px-2 py-0.5 rounded text-xs border border-slate-600',
              method.method_type === 'foundation model' && 'bg-blue-500/10 text-blue-400',
              method.method_type === 'hybrid' && 'bg-purple-500/10 text-purple-400',
              method.method_type === 'algorithmic' && 'bg-orange-500/10 text-orange-400'
            )}>
              {method.method_type}
            </span>
          </div>
          <p className="text-sm text-slate-400">{method.description}</p>
        </div>
        <ChevronRight className="w-5 h-5 text-slate-600 group-hover:text-slate-400 transition-colors" />
      </div>
    </Card>
  )
}

// =============================================================================
// Prompt Detail Modal
// =============================================================================

function MethodDetailModal({
  methodType,
  isOpen,
  onClose
}: {
  methodType: string | null
  isOpen: boolean
  onClose: () => void
}) {
  const { data: method, isLoading } = useQuery({
    queryKey: ['method', methodType],
    queryFn: () => methodsApi.get(methodType!),
    enabled: !!methodType && isOpen,
  })

  // Determine the default active tab based on available content
  const getDefaultTab = (method: any): 'system' | 'user' | 'output' | 'behavior' => {
    if (method?.system_prompt?.trim()) return 'system'
    if (method?.user_prompt?.trim()) return 'user'
    if (method?.output_format || method?.example_output) return 'output'
    if (method?.example_behavior?.trim()) return 'behavior'
    return 'system' // fallback
  }

  const [activeTab, setActiveTab] = useState<'system' | 'user' | 'output' | 'behavior'>(getDefaultTab(method))

  if (!methodType) return null

  const colors = getPlannerColors(methodType)

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`${method?.name || 'Loading...'}`} size="xl">
      {method && (
        <div className="flex items-center gap-2 mb-4">
          <span className={cn('px-2 py-0.5 rounded text-xs border border-slate-600',
            method.method_type === 'foundation model' && 'bg-blue-500/10 text-blue-400',
            method.method_type === 'hybrid' && 'bg-purple-500/10 text-purple-400',
            method.method_type === 'algorithmic' && 'bg-orange-500/10 text-orange-400'
          )}>
            {method.method_type}
          </span>
        </div>
      )}
      {isLoading ? (
        <div className="p-8 text-center text-slate-400">Loading method details...</div>
      ) : method ? (
        <div className="space-y-4">
          {/* Header with description */}
          <div className="space-y-2">
            <p className="text-sm text-slate-300">{method.description}</p>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 border-b border-slate-800 pb-2 overflow-x-auto">
            {method.system_prompt && method.system_prompt.trim() && (
              <button
                onClick={() => setActiveTab('system')}
                className={cn(
                  'px-4 py-2 rounded-t text-sm font-medium transition-colors whitespace-nowrap',
                  activeTab === 'system'
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-500 hover:text-slate-300'
                )}
              >
                System Prompt
              </button>
            )}
            {method.user_prompt && method.user_prompt.trim() && (
              <button
                onClick={() => setActiveTab('user')}
                className={cn(
                  'px-4 py-2 rounded-t text-sm font-medium transition-colors whitespace-nowrap',
                  activeTab === 'user'
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-500 hover:text-slate-300'
                )}
              >
                User Prompt
              </button>
            )}
            {(method.output_format || method.example_output) && (
              <button
                onClick={() => setActiveTab('output')}
                className={cn(
                  'px-4 py-2 rounded-t text-sm font-medium transition-colors whitespace-nowrap',
                  activeTab === 'output'
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-500 hover:text-slate-300'
                )}
              >
                Output Format
              </button>
            )}
            {method.example_behavior && method.example_behavior.trim() && (
              <button
                onClick={() => setActiveTab('behavior')}
                className={cn(
                  'px-4 py-2 rounded-t text-sm font-medium transition-colors whitespace-nowrap',
                  activeTab === 'behavior'
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-500 hover:text-slate-300'
                )}
              >
                Behavior
              </button>
            )}
          </div>

          {/* Content */}
          <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 max-h-[400px] overflow-y-auto">
            {activeTab === 'output' ? (
              <div className="space-y-4">
                {method.output_format && (
                  <div>
                    <h4 className="text-xs font-medium text-slate-500 uppercase mb-2">Format Description</h4>
                    <p className="text-sm text-slate-300">{method.output_format}</p>
                  </div>
                )}
                {method.example_output && (
                  <div>
                    <h4 className="text-xs font-medium text-slate-500 uppercase mb-2">Example Output</h4>
                    <pre className="text-sm text-emerald-300 whitespace-pre-wrap font-mono leading-relaxed bg-slate-950 p-3 rounded border border-slate-800">
                      {method.example_output}
                    </pre>
                  </div>
                )}
                {!method.output_format && !method.example_output && (
                  <p className="text-sm text-slate-400">No output format information available.</p>
                )}
              </div>
            ) : activeTab === 'behavior' ? (
              <div>
                <h4 className="text-xs font-medium text-slate-500 uppercase mb-2">Example Behavior</h4>
                <pre className="text-sm text-slate-300 whitespace-pre-wrap font-mono leading-relaxed">
                  {method.example_behavior}
                </pre>
              </div>
            ) : (
              <pre className="text-sm text-slate-300 whitespace-pre-wrap font-mono leading-relaxed">
                {activeTab === 'system' ? method.system_prompt : method.user_prompt}
              </pre>
            )}
          </div>

          {/* Prompts info (if available) */}
          {method.prompts && method.prompts.length > 0 && (
            <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="w-4 h-4 text-cyber-400" />
                <span className="text-sm font-medium text-slate-300">Prompts Used</span>
              </div>
              <div className="space-y-2">
                {method.prompts.map((prompt, index) => (
                  <div key={index} className="text-sm">
                    <span className="text-cyber-400 font-medium">{prompt.type}:</span>
                    <span className="text-slate-300 ml-2">{prompt.description}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Template Variables (only for user prompt) */}
          {activeTab === 'user' && method.variables.length > 0 && (
            <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
              <div className="flex items-center gap-2 mb-3">
                <Variable className="w-4 h-4 text-cyber-400" />
                <span className="text-sm font-medium text-slate-300">Template Variables</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {method.variables.map((variable) => (
                  <code
                    key={variable}
                    className="px-2 py-1 bg-cyber-500/10 border border-cyber-500/30 rounded text-xs text-cyber-300 font-mono"
                  >
                    {`{${variable}}`}
                  </code>
                ))}
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="flex justify-end pt-4 border-t border-slate-800">
            <Button variant="secondary" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      ) : (
        <div className="p-8 text-center text-red-400">Failed to load method details</div>
      )}
    </Modal>
  )
}

// =============================================================================
// Main Allocators Page
// =============================================================================

export function Allocators() {
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null)

  const { data: methods = [], isLoading } = useQuery({
    queryKey: ['allocators'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'allocator')),
  })

  if (isLoading) {
    return <div className="text-slate-400">Loading...</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Allocators</h1>
        <p className="text-slate-400">
          View the allocation methods available in the system. Each allocator defines how tasks are assigned to robots.
        </p>
      </div>

      {/* Info Banner */}
      <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
        <p className="text-sm text-slate-300">
          <strong className="text-white">Allocator Types:</strong> Some use{' '}
          <span className="text-cyber-400">LLM reasoning</span> for intelligent task assignment, others use{' '}
          <span className="text-purple-400">hybrid approaches</span> combining algorithmic orchestration with LLM assistance, and some use{' '}
          <span className="text-orange-400">mathematical optimization</span> for balanced workloads.
        </p>
      </div>

      {/* Methods Grid */}
      {methods.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {methods.map((method) => (
            <MethodCard
              key={`${method.category}-${method.type}`}
              method={method}
              onClick={() => setSelectedMethod(method.type)}
            />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<FileText className="w-8 h-8" />}
          title="No allocators found"
          description="Allocator configuration files may be missing from the allocators types directory."
        />
      )}

      {/* Detail Modal */}
      <MethodDetailModal
        methodType={selectedMethod}
        isOpen={!!selectedMethod}
        onClose={() => setSelectedMethod(null)}
      />
    </div>
  )
}
