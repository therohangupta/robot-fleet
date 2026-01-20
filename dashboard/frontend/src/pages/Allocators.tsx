import { useState, useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Cog, FileText, ChevronRight, Variable, Search, SortAsc, SortDesc, Filter } from 'lucide-react'
import { Card } from '../components/common/Card'
import { Button } from '../components/common/Button'
import { Modal } from '../components/common/Modal'
import { EmptyState } from '../components/common/EmptyState'
import { methodsApi, type MethodSummary } from '../lib/api'
import { cn } from '../lib/utils'
import { MethodDetailModal } from './Planners'

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
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-white">{method.name}</h3>
          </div>
          <p className="text-sm text-slate-400">{method.description}</p>
        </div>
        <span className={cn('px-2 py-0.5 rounded text-xs border border-slate-600 ml-4',
          method.method_type === 'foundation model' && 'bg-blue-500/10 text-blue-400',
          method.method_type === 'hybrid' && 'bg-purple-500/10 text-purple-400',
          method.method_type === 'algorithmic' && 'bg-orange-500/10 text-orange-400'
        )}>
          {method.method_type}
        </span>
      </div>
    </Card>
  )
}


// =============================================================================
// Main Allocators Page
// =============================================================================

export function Allocators() {
  const [selectedMethod, setSelectedMethod] = useState<number | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'name' | 'type'>('name')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  // Handle URL query parameters for method modal
  useEffect(() => {
    const method = searchParams.get('method')
    if (method) {
      const methodId = parseInt(method, 10)
      if (!isNaN(methodId)) {
        setSelectedMethod(methodId)
      }
    }
  }, [searchParams])

  const { data: allMethods = [], isLoading } = useQuery({
    queryKey: ['allocators'],
    queryFn: () => methodsApi.list().then(methods => methods.filter(m => m.category === 'allocator')),
  })

  // Filter and sort methods based on search, type filter, and sort criteria
  const methods = useMemo(() => {
    let filtered = allMethods

    // Apply search filter first
    if (searchTerm.trim()) {
      const searchLower = searchTerm.toLowerCase()
      filtered = filtered.filter(method => {
        const nameMatch = method.name?.toLowerCase().includes(searchLower)
        const descMatch = method.description?.toLowerCase().includes(searchLower)
        const typeMatch = method.type?.toLowerCase().includes(searchLower)

        return nameMatch || descMatch || typeMatch
      })
    }

    // Apply type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter(method => method.method_type === typeFilter)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue: string, bValue: string

      if (sortBy === 'name') {
        aValue = a.name || ''
        bValue = b.name || ''
      } else {
        aValue = a.method_type || ''
        bValue = b.method_type || ''
      }

      const comparison = aValue.localeCompare(bValue)
      return sortOrder === 'asc' ? comparison : -comparison
    })

    return filtered
  }, [allMethods, searchTerm, typeFilter, sortBy, sortOrder])

  // Get unique method types for filter buttons
  const availableTypes = useMemo(() => {
    const types = new Set(allMethods.map(method => method.method_type))
    return Array.from(types).sort()
  }, [allMethods])

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

      {/* Filters and Search */}
      <div className="space-y-4">
        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search allocators by name, description, or type..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyber-500"
          />
        </div>

        {/* Type Filters and Sort Controls */}
        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
          {/* Type Filter Buttons */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setTypeFilter('all')}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                typeFilter === 'all'
                  ? 'bg-cyber-500 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-300'
              )}
            >
              <Filter className="w-3.5 h-3.5 inline mr-1" />
              All Types ({allMethods.length})
            </button>
            {availableTypes.map(methodType => {
              const count = allMethods.filter(method => method.method_type === methodType).length
              return (
                <button
                  key={methodType}
                  onClick={() => setTypeFilter(methodType)}
                  className={cn(
                    'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                    typeFilter === methodType
                      ? 'bg-cyber-500 text-white'
                      : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-300'
                  )}
                >
                  {methodType} ({count})
                </button>
              )
            })}
          </div>

          {/* Sort Controls */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-400">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'name' | 'method_type')}
              className="px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-md text-white text-sm focus:outline-none focus:border-cyber-500"
            >
              <option value="name">Name</option>
              <option value="method_type">Method Type</option>
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-1.5 bg-slate-800 border border-slate-700 rounded-md text-slate-400 hover:text-white hover:border-slate-600 transition-colors"
              title={`Sort ${sortOrder === 'asc' ? 'descending' : 'ascending'}`}
            >
              {sortOrder === 'asc' ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Results Summary */}
        {(searchTerm || typeFilter !== 'all') && (
          <div className="text-sm text-slate-400">
            Showing {methods.length} of {allMethods.length} allocators
            {searchTerm && ` matching "${searchTerm}"`}
            {typeFilter !== 'all' && ` of type "${typeFilter}"`}
          </div>
        )}
      </div>

      {/* Methods Grid */}
      {methods.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {methods.map((method) => (
            <MethodCard
              key={`${method.category}-${method.type}`}
              method={method}
              onClick={() => navigate(`/allocators?method=${method.id}`)}
            />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<Cog className="w-8 h-8" />}
          title="No allocators found"
          description="Allocator configuration files may be missing from the allocators types directory."
        />
      )}

      {/* Detail Modal */}
      <MethodDetailModal
        methodId={selectedMethod}
        methodType="allocator"
        isOpen={!!selectedMethod}
        onClose={() => {
          setSelectedMethod(null)
          // Clear URL parameters when modal closes
          navigate('/allocators', { replace: true })
        }}
      />
    </div>
  )
}
