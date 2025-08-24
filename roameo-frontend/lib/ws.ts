import { WS_URL, type WsEvent } from "./types"

type WsCallbacks = {
  onOpen?: () => void
  onClose?: (ev: CloseEvent) => void
  onError?: (ev: Event) => void
}

export function connectWs(sessionId: string, onEvent: (evt: WsEvent) => void, cbs: WsCallbacks = {}) {
  const url = `${WS_URL}?sessionId=${encodeURIComponent(sessionId)}`
  const ws = new WebSocket(url)
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
