import { WS_URL, type WsEvent } from "./types"

type WsCallbacks = {
  onOpen?: () => void
  onClose?: (ev: CloseEvent) => void
  onError?: (ev: Event) => void
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

  const ws = new WebSocket(url)
  
  // Add auth header if available
  getAuthToken().then(token => {
    if (token && ws.readyState === WebSocket.CONNECTING) {
      // Note: WebSocket doesn't support custom headers after creation
      // We'll need to send auth via message after connection
      ws.addEventListener('open', () => {
        ws.send(JSON.stringify({ type: 'auth', token }))
      }, { once: true })
    }
  })

  ws.onopen = () => cbs.onOpen?.()
  ws.onclose = (ev) => cbs.onClose?.(ev)
  ws.onerror = (ev) => cbs.onError?.(ev)
  ws.onmessage = (m) => {
    try {
      const evt = JSON.parse(m.data as string) as WsEvent
      onEvent(evt)
    } catch (e) {
      console.error("WS parse error", e)
    }
  }
  return ws
}
