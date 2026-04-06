import type { Task } from '../models/types'

export type GatewayRealtimeMessage =
  | { type: 'connected'; message: string }
  | { type: 'invalidate'; queries: string[]; timestamp: number }
  | { type: 'tasks_update'; plan_id: number; tasks: Task[] }
  // Future-proof: unknown messages pass through as generic objects
  | ({ type: string } & Record<string, any>)

