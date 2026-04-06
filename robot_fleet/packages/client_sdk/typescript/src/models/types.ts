// Shared types for any TypeScript client (web/mobile/node).

export type TaskStatus = 'unknown' | 'pending' | 'in_progress' | 'completed' | 'cancelled' | 'failed'
export type RobotState = 'unknown' | 'registered' | 'deploying' | 'running' | 'error' | 'stopped'

export interface TaskServerInfo {
  host: string
  port: number
}

export interface ContainerInfo {
  container_id?: string
  image?: string
  host?: string
  port?: number
}

export interface Robot {
  robot_id: string
  robot_type: string
  description?: string
  capabilities: string[]
  status?: RobotState
  task_server_info?: TaskServerInfo
  container?: ContainerInfo
  task_ids: number[]
}

export type PlanExecutionStatus = 'not_executed' | 'executing' | 'completed' | 'failed'

export interface RobotAllocationsPlanSummary {
  plan_id: number
  goal_ids: number[]
  task_count: number
  status: PlanExecutionStatus
  name: string
  description: string
}

export interface RobotAllocationsResponse {
  robot_id: string
  plans_count: number
  goals_count: number
  tasks_count: number
  plans: RobotAllocationsPlanSummary[]
  goals: number[]
}

export interface Goal {
  goal_id: number
  description: string
  task_ids: number[]
}

export interface Task {
  task_id: number
  description: string
  goal_id?: number
  plan_id?: number
  robot_id?: string
  robot_type?: string
  dependency_task_ids: number[]
  status: TaskStatus
  result?: string | null
}

export interface Plan {
  plan_id: number
  name: string
  description: string
  planning_strategy: number
  allocation_strategy: number
  task_ids: number[]
  goal_ids: number[]
  tasks?: Task[]
  planning_prompts?: Record<string, string>
  allocation_prompts?: Record<string, string>
  planning_artifacts?: Record<string, any>
  allocation_artifacts?: Record<string, any>
  server_logs?: string
  dag_structure?: Record<string, any>
  created_at?: string
  execution_status?: string
}

export interface WorldStatement {
  id: string
  statement: string
  created_at?: string
}

export interface Embodiment {
  name: string
  description: string
  capabilities: string[]
  default_port: number
  config_path: string
  container_image: string
}

export interface Strategy {
  value: string
  label: string
  description: string
}

export interface StrategiesResponse {
  planning: Strategy[]
  allocation: Strategy[]
}

// Methods (planners/allocators) metadata
export interface MethodSummary {
  category: string
  type: string
  id?: number
  name: string
  description: string
  method_type: string
  output_format?: string
  example_output?: string
  example_behavior?: string
  prompts: Array<{
    type: string
    description: string
  }>
}

export interface MethodDetail extends MethodSummary {
  system_prompt: string
  user_prompt: string
  variables: string[]
}

export interface RobotInstanceCreateRequest {
  config_path: string
  robot_id: string
  host: string
  port: number
}

export interface GoalCreateRequest {
  description: string
}

export interface PlanCreateRequest {
  planning_strategy: number
  allocation_strategy: number
  goal_ids: number[]
  name: string
  description: string
}

export interface ManualTaskDefinition {
  temp_id: string
  description: string
  robot_id?: string
  robot_type?: string
  depends_on: string[]
  goal_id?: number
}

export interface ManualPlanCreateRequest {
  name: string
  description: string
  tasks: ManualTaskDefinition[]
}

export type PlanAllocationStatus = 'empty' | 'unallocated' | 'partially_allocated' | 'fully_allocated'

export interface PlanStatus {
  plan_id: number
  status: PlanAllocationStatus
  total_tasks: number
  allocated_tasks: number
  unallocated_task_ids: number[]
  is_executable: boolean
}

