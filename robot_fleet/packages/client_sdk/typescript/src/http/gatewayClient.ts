import { fetchApi } from './fetchApi'
import type {
  Embodiment,
  Goal,
  GoalCreateRequest,
  Plan,
  PlanCreateRequest,
  PlanStatus,
  Robot,
  RobotAllocationsResponse,
  RobotInstanceCreateRequest,
  StrategiesResponse,
  Task,
  WorldStatement,
  WorldStatementCreateRequest,
  ManualPlanCreateRequest,
  MethodSummary,
  MethodDetail,
} from '../models/types'

export interface GatewayClientOptions {
  /** Example: "http://localhost:8000" (no trailing slash). Empty string for same-origin. */
  baseUrl?: string
  /** If provided, sent as Authorization: Bearer <token> */
  bearerToken?: string
}

export class GatewayClient {
  private baseUrl: string
  private bearerToken?: string

  constructor(opts?: GatewayClientOptions) {
    this.baseUrl = opts?.baseUrl ?? ''
    this.bearerToken = opts?.bearerToken
  }

  private headers(): Record<string, string> {
    return this.bearerToken ? { Authorization: `Bearer ${this.bearerToken}` } : {}
  }

  // Robots
  robots = {
    list: (filter = 'all') => fetchApi<Robot[]>(`/api/robots?filter=${filter}`, { baseUrl: this.baseUrl, headers: this.headers() }),
    get: (robotId: string) => fetchApi<Robot>(`/api/robots/${robotId}`, { baseUrl: this.baseUrl, headers: this.headers() }),
    getAllocations: (robotId: string) =>
      fetchApi<RobotAllocationsResponse>(`/api/robots/${robotId}/allocations`, { baseUrl: this.baseUrl, headers: this.headers() }),
    register: (data: RobotInstanceCreateRequest) =>
      fetchApi<Robot>(`/api/robots/register`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify(data),
      }),
    unregister: (robotId: string) =>
      fetchApi<{ success: boolean }>(`/api/robots/${robotId}`, {
        baseUrl: this.baseUrl,
        method: 'DELETE',
        headers: this.headers(),
      }),
  }

  // Goals
  goals = {
    list: () => fetchApi<Goal[]>(`/api/goals`, { baseUrl: this.baseUrl, headers: this.headers() }),
    get: (goalId: number) => fetchApi<Goal>(`/api/goals/${goalId}`, { baseUrl: this.baseUrl, headers: this.headers() }),
    create: (data: GoalCreateRequest) =>
      fetchApi<Goal>(`/api/goals`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify(data),
      }),
    delete: (goalId: number) =>
      fetchApi<{ success: boolean }>(`/api/goals/${goalId}`, {
        baseUrl: this.baseUrl,
        method: 'DELETE',
        headers: this.headers(),
      }),
  }

  // Plans
  plans = {
    list: () => fetchApi<Plan[]>(`/api/plans`, { baseUrl: this.baseUrl, headers: this.headers() }),
    get: (planId: number) => fetchApi<Plan>(`/api/plans/${planId}`, { baseUrl: this.baseUrl, headers: this.headers() }),
    create: (data: PlanCreateRequest) =>
      fetchApi<Plan>(`/api/plans`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify(data),
      }),
    createManual: (data: ManualPlanCreateRequest) =>
      fetchApi<Plan>(`/api/plans/manual`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify(data),
      }),
    allocate: (planId: number, allocationStrategy: string) =>
      fetchApi<Plan>(`/api/plans/${planId}/allocate`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify({ allocation_strategy: allocationStrategy }),
      }),
    getStatus: (planId: number) => fetchApi<PlanStatus>(`/api/plans/${planId}/status`, { baseUrl: this.baseUrl, headers: this.headers() }),
    start: (planId: number) =>
      fetchApi<{ success: boolean }>(`/api/plans/${planId}/start`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
      }),
    copy: (planId: number, data: { name: string; description: string }) =>
      fetchApi<Plan>(`/api/plans/${planId}/copy`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify(data),
      }),
    update: (planId: number, data: { name: string; description: string }) =>
      fetchApi<Plan>(`/api/plans/${planId}`, {
        baseUrl: this.baseUrl,
        method: 'PUT',
        headers: this.headers(),
        body: JSON.stringify(data),
      }),
    delete: (planId: number) =>
      fetchApi<{ success: boolean }>(`/api/plans/${planId}`, {
        baseUrl: this.baseUrl,
        method: 'DELETE',
        headers: this.headers(),
      }),
  }

  // Tasks
  tasks = {
    list: (params?: { plan_id?: number; goal_id?: number; robot_id?: string }) => {
      const searchParams = new URLSearchParams()
      if (params?.plan_id) searchParams.set('plan_id', String(params.plan_id))
      if (params?.goal_id) searchParams.set('goal_id', String(params.goal_id))
      if (params?.robot_id) searchParams.set('robot_id', params.robot_id)
      const query = searchParams.toString()
      return fetchApi<Task[]>(`/api/tasks${query ? `?${query}` : ''}`, { baseUrl: this.baseUrl, headers: this.headers() })
    },
    get: (taskId: number) => fetchApi<Task>(`/api/tasks/${taskId}`, { baseUrl: this.baseUrl, headers: this.headers() }),
    create: (data: {
      description: string
      goal_id: number
      plan_id?: number
      robot_id?: string | null
      robot_type?: string | null
      dependency_task_ids?: number[]
    }) =>
      fetchApi<Task>(`/api/tasks`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify({
          description: data.description,
          goal_id: data.goal_id,
          plan_id: data.plan_id ?? null,
          robot_id: data.robot_id ?? null,
          robot_type: data.robot_type ?? null,
          dependency_task_ids: data.dependency_task_ids ?? [],
        }),
      }),
    update: (
      taskId: number,
      data: {
        description?: string
        goal_id?: number
        robot_id?: string
        update_dependency_task_ids?: boolean
        dependency_task_ids?: number[]
      }
    ) =>
      fetchApi<Task>(`/api/tasks/${taskId}`, {
        baseUrl: this.baseUrl,
        method: 'PATCH',
        headers: this.headers(),
        body: JSON.stringify({
          ...data,
          dependency_task_ids: data.dependency_task_ids ?? [],
          update_dependency_task_ids: data.update_dependency_task_ids ?? false,
        }),
      }),
    delete: (taskId: number) =>
      fetchApi<{ success: boolean; deleted_task_id: number; updated_task_ids: number[] }>(`/api/tasks/${taskId}`, {
        baseUrl: this.baseUrl,
        method: 'DELETE',
        headers: this.headers(),
      }),
  }

  // World
  world = {
    list: () => fetchApi<WorldStatement[]>(`/api/world`, { baseUrl: this.baseUrl, headers: this.headers() }),
    add: (data: WorldStatementCreateRequest) =>
      fetchApi<WorldStatement>(`/api/world`, {
        baseUrl: this.baseUrl,
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify(data),
      }),
    delete: (statementId: string) =>
      fetchApi<{ success: boolean }>(`/api/world/${statementId}`, {
        baseUrl: this.baseUrl,
        method: 'DELETE',
        headers: this.headers(),
      }),
  }

  // Embodiments
  embodiments = {
    list: () => fetchApi<Embodiment[]>(`/api/embodiments`, { baseUrl: this.baseUrl, headers: this.headers() }),
    get: (name: string) => fetchApi<Embodiment>(`/api/embodiments/${name}`, { baseUrl: this.baseUrl, headers: this.headers() }),
    suggestPort: (basePort = 8001) =>
      fetchApi<{ suggested_port: number }>(`/api/ports/suggest?base_port=${basePort}`, { baseUrl: this.baseUrl, headers: this.headers() }),
    usedPorts: () => fetchApi<{ used_ports: number[] }>(`/api/ports/used`, { baseUrl: this.baseUrl, headers: this.headers() }),
  }

  strategies = {
    get: () => fetchApi<StrategiesResponse>(`/api/strategies`, { baseUrl: this.baseUrl, headers: this.headers() }),
  }

  methods = {
    list: () => fetchApi<MethodSummary[]>(`/api/methods`, { baseUrl: this.baseUrl, headers: this.headers() }),
    get: (methodId: number, methodType?: 'planner' | 'allocator') => {
      const url = methodType ? `/api/methods/${methodId}?category=${methodType}` : `/api/methods/${methodId}`
      return fetchApi<MethodDetail>(url, { baseUrl: this.baseUrl, headers: this.headers() })
    },
  }
}

