import React from 'react'
import { Button } from './Button'

let graphvizInstance: any = null
let graphvizLoading: Promise<any> | null = null

function getGraphviz(): Promise<any> {
  if (graphvizInstance) return Promise.resolve(graphvizInstance)
  if (graphvizLoading) return graphvizLoading
  graphvizLoading = import('@hpcc-js/wasm-graphviz').then(async ({ Graphviz }) => {
    graphvizInstance = await Graphviz.load()
    return graphvizInstance
  })
  return graphvizLoading
}

interface DAGVisualizationProps {
  tasks: any[]
  height?: string
}

function generateDot(tasks: any[]): string {
  let dot = `digraph G {
  rankdir=LR;
  bgcolor="#ffffff";
  node [shape=plaintext, fontname="Arial"];
  edge [color="#334155",penwidth=5.0, arrowhead=vee, arrowsize=4.0, headclip=true, tailclip=true];
  graph [splines=spline, nodesep=4.0, ranksep=4.0];
`
  tasks.forEach(task => {
    let statusColors = { border: '#1e293b', bg: '#ffffff', text: '#0f172a', headerBg: '#f1f5f9' }

    switch ((task.status || 'pending').toLowerCase()) {
      case 'completed':
        statusColors = { border: '#047857', bg: '#d1fae5', text: '#064e3b', headerBg: '#a7f3d0' }
        break
      case 'running': case 'executing': case 'in_progress':
        statusColors = { border: '#d97706', bg: '#fef3c7', text: '#92400e', headerBg: '#fde68a' }
        break
      case 'failed': case 'error':
        statusColors = { border: '#dc2626', bg: '#fee2e2', text: '#991b1b', headerBg: '#fecaca' }
        break
      case 'pending': default:
        statusColors = { border: '#64748b', bg: '#f8fafc', text: '#374151', headerBg: '#f3f4f6' }
        break
    }

    const taskId = task.task_id.toString()
    const description = task.description.replace(/"/g, '\\"').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    const robotType = task.robot_type || 'unknown'
    const robotId = task.robot_id || 'unassigned'

    const htmlLabel = `<TABLE BORDER="3" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" BGCOLOR="${statusColors.bg}" COLOR="${statusColors.border}" STYLE="ROUNDED">
  <TR>
    <TD ALIGN="CENTER" BGCOLOR="${statusColors.headerBg}" CELLPADDING="18">
      <FONT COLOR="#1d4ed8" FACE="Arial" POINT-SIZE="50"><B>Task ${taskId}</B></FONT>
    </TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" CELLPADDING="20">
      <FONT COLOR="${statusColors.text}" FACE="Arial" POINT-SIZE="44"><b>${description}</b></FONT>
    </TD>
  </TR>
  <TR>
    <TD ALIGN="CENTER" CELLPADDING="16">
      <TABLE BORDER="0" CELLSPACING="8">
        <TR>
          <TD BGCOLOR="#7c3aed" CELLPADDING="14">
            <FONT COLOR="#ffffff" FACE="Arial" POINT-SIZE="40"><B>${robotType}</B></FONT>
          </TD>
        </TR>
        <TR>
          <TD BGCOLOR="#0d9488" CELLPADDING="14">
            <FONT COLOR="#ffffff" FACE="Arial" POINT-SIZE="40"><B>${robotId}</B></FONT>
          </TD>
        </TR>
      </TABLE>
    </TD>
  </TR>
</TABLE>`

    dot += `  ${task.task_id} [label=<${htmlLabel}>];\n`
  })

  tasks.forEach(task => {
    if (task.dependency_task_ids && task.dependency_task_ids.length > 0) {
      task.dependency_task_ids.forEach((depId: number) => {
        dot += `  ${depId} -> ${task.task_id} [color="#374151", penwidth="3"];\n`
      })
    }
  })

  dot += '}'
  return dot
}

export function DAGVisualization({ tasks, height = '600px' }: DAGVisualizationProps) {
  const [svgContent, setSvgContent] = React.useState<string>('')
  const [isGenerating, setIsGenerating] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [zoom, setZoom] = React.useState(0.6)
  const [pan, setPan] = React.useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = React.useState(false)
  const [lastMousePos, setLastMousePos] = React.useState({ x: 0, y: 0 })
  const containerRef = React.useRef<HTMLDivElement>(null)

  const resetView = () => {
    setZoom(0.6)
    setPan({ x: 0, y: 0 })
  }

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'r' || e.key === 'R') resetView()
      else if (e.key === '+' || e.key === '=') setZoom(prev => Math.min(3, prev * 1.2))
      else if (e.key === '-') setZoom(prev => Math.max(0.1, prev * 0.8))
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true)
      setLastMousePos({ x: e.clientX, y: e.clientY })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan(prev => ({
        x: prev.x + e.clientX - lastMousePos.x,
        y: prev.y + e.clientY - lastMousePos.y,
      }))
      setLastMousePos({ x: e.clientX, y: e.clientY })
    }
  }

  const handleMouseUp = () => setIsDragging(false)

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const factor = e.deltaY > 0 ? 0.9 : 1.1
    setZoom(prev => Math.min(3, Math.max(0.1, prev * factor)))
  }

  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      setIsDragging(true)
      setLastMousePos({ x: e.touches[0].clientX, y: e.touches[0].clientY })
    }
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    if (isDragging && e.touches.length === 1) {
      e.preventDefault()
      setPan(prev => ({
        x: prev.x + e.touches[0].clientX - lastMousePos.x,
        y: prev.y + e.touches[0].clientY - lastMousePos.y,
      }))
      setLastMousePos({ x: e.touches[0].clientX, y: e.touches[0].clientY })
    }
  }

  const handleTouchEnd = () => setIsDragging(false)

  const taskSignature = React.useMemo(
    () => tasks.map(t => `${t.task_id}:${t.status}`).join(','),
    [tasks]
  )

  React.useEffect(() => {
    if (!tasks || tasks.length === 0) {
      setSvgContent('')
      setError(null)
      setIsGenerating(false)
      return
    }

    setIsGenerating(true)
    setError(null)

    const dotSource = generateDot(tasks)

    getGraphviz().then(gv => {
      try {
        const svg = gv.dot(dotSource, 'svg')
        if (svg && svg.length > 100 && svg.includes('<svg')) {
          setSvgContent(svg)
        } else {
          setError('Generated SVG is invalid')
        }
      } catch (renderError: any) {
        setError(`Graphviz render failed: ${renderError.message}`)
      }
      setIsGenerating(false)
    }).catch((err: any) => {
      setError(`Graphviz load failed: ${err.message}`)
      setIsGenerating(false)
    })
  }, [taskSignature])

  if (!tasks || tasks.length === 0) {
    return (
      <div className="text-center py-8 text-[var(--color-text-secondary)] bg-surface-overlay/50 rounded-lg">
        No tasks to visualize
      </div>
    )
  }

  if (isGenerating && !svgContent) {
    return (
      <div className="text-center py-8 text-[var(--color-text-secondary)] bg-surface-overlay/50 rounded-lg">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-border mx-auto mb-2"></div>
        Generating DAG visualization...
      </div>
    )
  }

  if (error && !svgContent) {
    return (
      <div className="space-y-4">
        <div className="text-sm text-red-400 text-center">Graphviz failed — showing text fallback</div>
        <div className="bg-surface border border-border rounded-lg p-4 overflow-auto max-h-96">
          <div className="font-mono text-sm text-[var(--color-text)] space-y-2">
            {tasks.map(task => (
              <div key={task.task_id} className="flex items-start space-x-4">
                <div className="text-blue-400 font-bold min-w-[3rem]">T{task.task_id}</div>
                <div className="flex-1">
                  <div className="text-white">{task.description}</div>
                  {task.dependency_task_ids?.length > 0 && (
                    <div className="text-[var(--color-text-secondary)] text-xs mt-1">
                      Depends on: {task.dependency_task_ids.join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="text-xs text-[var(--color-text-muted)] text-center">Error: {error}</div>
      </div>
    )
  }

  return (
    <div className="space-y-3 min-w-0 overflow-hidden">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">Completed</span>
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-500/15 text-amber-400 border border-amber-500/20">Running</span>
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-red-500/15 text-red-400 border border-red-500/20">Failed</span>
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-slate-500/15 text-slate-400 border border-slate-500/20">Pending</span>
          <span className="text-[10px] text-[var(--color-text-muted)] ml-1">Arrows = dependencies</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-[var(--color-text-muted)]">{Math.round(zoom * 100)}%</span>
          <Button variant="secondary" size="sm" onClick={resetView} className="text-xs">
            Reset
          </Button>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative bg-white border border-border rounded-lg overflow-hidden shadow-lg"
        style={{ height, cursor: isDragging ? 'grabbing' : 'grab' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <div
          className="absolute inset-0 flex items-center justify-center [&>svg]:max-w-none"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.1s ease-out',
          }}
          dangerouslySetInnerHTML={{ __html: svgContent }}
        />
      </div>
    </div>
  )
}
