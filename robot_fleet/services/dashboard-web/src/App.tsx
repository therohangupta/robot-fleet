import { Routes, Route, Navigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { Layout } from './components/layout/Layout'
import { Dashboard } from './pages/Dashboard'
import { Robots } from './pages/Robots'
import { Goals } from './pages/Goals'
import { Plans } from './pages/Plans'
import PlanDetails from './pages/PlanDetails'
import { Execution } from './pages/Execution'
import { World } from './pages/World'
import { Planners } from './pages/Planners'
import { Allocators } from './pages/Allocators'
import { methodsApi, useRealtimeUpdates } from './lib/api'
import { setMethodData } from './lib/utils'

function App() {
  useRealtimeUpdates()
  const [methodsLoaded, setMethodsLoaded] = useState(false)

  // Load method data synchronously on app startup - block rendering until loaded
  useEffect(() => {
    const loadMethodData = async () => {
      try {
        console.log('Loading method data from YAML files...')
        const methods = await methodsApi.list()
        const planners = methods.filter(m => m.category === 'planner')
        const allocators = methods.filter(m => m.category === 'allocator')
        setMethodData(planners, allocators)
        console.log('Method data loaded from YAML files:', {
          planners: planners.length,
          allocators: allocators.length,
          plannerNames: planners.map(p => `${p.id}: ${p.name}`),
          allocatorNames: allocators.map(a => `${a.id}: ${a.name}`)
        })
        setMethodsLoaded(true)
      } catch (error) {
        console.error('Failed to load method data from YAML files:', error)
        // Still set loaded to true to avoid infinite loading, but log error
        setMethodsLoaded(true)
      }
    }

    loadMethodData()
  }, [])

  // Show loading screen until method data is loaded
  if (!methodsLoaded) {
    return (
      <div className="min-h-screen bg-[var(--color-bg)] flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyber-500 mx-auto mb-4"></div>
          <p className="text-[var(--color-text-secondary)]">Loading method configurations...</p>
        </div>
      </div>
    )
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/robots" element={<Robots />} />
        <Route path="/goals" element={<Goals />} />
        <Route path="/plans" element={<Plans />} />
        <Route path="/plans/:planId" element={<PlanDetails />} />
        <Route path="/plans/:planId/execute" element={<Execution />} />
        <Route path="/world" element={<World />} />
        <Route path="/planners" element={<Planners />} />
        <Route path="/allocators" element={<Allocators />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}

export default App
