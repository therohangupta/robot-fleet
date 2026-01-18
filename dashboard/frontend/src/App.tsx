import { Routes, Route, Navigate } from 'react-router-dom'
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

function App() {
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
