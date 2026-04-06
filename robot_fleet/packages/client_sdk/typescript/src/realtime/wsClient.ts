import type { GatewayRealtimeMessage } from './events'

export interface GatewayRealtimeClientOptions {
  /** Example: "ws://localhost:8000" (no trailing slash). Empty string means relative WS URL. */
  wsBaseUrl?: string
}

export class GatewayRealtimeClient {
  private wsBaseUrl: string

  constructor(opts?: GatewayRealtimeClientOptions) {
    this.wsBaseUrl = opts?.wsBaseUrl ?? ''
  }

  connectGlobalUpdates(onMessage: (msg: GatewayRealtimeMessage) => void): WebSocket {
    const url = `${this.wsBaseUrl}/ws/global-updates`
    const ws = new WebSocket(url)
    ws.onmessage = (ev) => {
      try {
        onMessage(JSON.parse(ev.data))
      } catch {
        // ignore
      }
    }
    return ws
  }

  connectPlanExecution(planId: number, onMessage: (msg: GatewayRealtimeMessage) => void): WebSocket {
    const url = `${this.wsBaseUrl}/ws/execution/${planId}`
    const ws = new WebSocket(url)
    ws.onmessage = (ev) => {
      try {
        onMessage(JSON.parse(ev.data))
      } catch {
        // ignore
      }
    }
    return ws
  }
}

