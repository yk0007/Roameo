import { WS_URL, type WsEvent } from "./types"

type WsCallbacks = {
  onOpen?: () => void
  onClose?: (ev: CloseEvent) => void
  onError?: (ev: Event) => void
  onHealthCheck?: (isHealthy: boolean) => void
}

interface WebSocketWithHealth extends WebSocket {
  _lastPing?: number
  _pingInterval?: number
  _healthCheckInterval?: number
}

export function connectWs(sessionId: string, onEvent: (evt: WsEvent) => void, cbs: WsCallbacks = {}) {
  const url = `${WS_URL}?sessionId=${encodeURIComponent(sessionId)}`
  
  // Get auth token for WebSocket connection
  const getAuthToken = async () => {
    try {
      const { supabase } = await import("@/lib/supabase/client")
      const { data: { session } } = await supabase.auth.getSession()
      return session?.access_token
    } catch {
      return null
    }
  }

  const ws = new WebSocket(url) as WebSocketWithHealth
  
  // Health check mechanism
  const setupHealthCheck = () => {
    // Send ping every 30 seconds
    ws._pingInterval = window.setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws._lastPing = Date.now()
        ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, 30000)
    
    // Check health every 5 seconds
    ws._healthCheckInterval = window.setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        const now = Date.now()
        const isHealthy = !ws._lastPing || (now - ws._lastPing) < 60000 // 1 minute timeout
        cbs.onHealthCheck?.(isHealthy)
        
        if (!isHealthy) {
          console.warn('WebSocket appears unhealthy, closing connection')
          ws.close(1000, 'Health check failed')
        }
      }
    }, 5000)
  }
  
  const cleanup = () => {
    if (ws._pingInterval) {
      clearInterval(ws._pingInterval)
      ws._pingInterval = undefined
    }
    if (ws._healthCheckInterval) {
      clearInterval(ws._healthCheckInterval)
      ws._healthCheckInterval = undefined
    }
  }
  
  // Add auth header if available
  getAuthToken().then(token => {
    if (token && ws.readyState === WebSocket.CONNECTING) {
      // Note: WebSocket doesn't support custom headers after creation
      // We'll need to send auth via message after connection
      ws.addEventListener('open', () => {
        ws.send(JSON.stringify({ type: 'auth', token }))
        setupHealthCheck()
      }, { once: true })
    } else {
      ws.addEventListener('open', () => {
        setupHealthCheck()
      }, { once: true })
    }
  })

  ws.onopen = () => {
    ws._lastPing = Date.now()
    cbs.onOpen?.()
  }
  
  ws.onclose = (ev) => {
    cleanup()
    cbs.onClose?.(ev)
  }
  
  ws.onerror = (ev) => {
    cleanup()
    cbs.onError?.(ev)
  }
  
  ws.onmessage = (m) => {
    try {
      const data = JSON.parse(m.data as string)
      
      // Handle pong response
      if (data.type === 'pong') {
        ws._lastPing = Date.now()
        return
      }
      
      // Handle regular events
      const evt = data as WsEvent
      onEvent(evt)
    } catch (e) {
      console.error("WS parse error", e)
    }
  }
  
  // Return enhanced WebSocket with cleanup method
  return Object.assign(ws, {
    closeWithCleanup: () => {
      cleanup()
      ws.close()
    }
  })
}
