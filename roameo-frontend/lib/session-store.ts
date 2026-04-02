import { create } from "zustand";
import type { ChatMessage, CanonicalSession, WsEvent } from "./types";

type SessionState = {
  snapshot?: CanonicalSession;
  streamingMessage?: ChatMessage;
  activeTurnId?: string;
  isStreaming: boolean;
  error?: string;
  reset: () => void;
  hydrate: (snapshot: CanonicalSession) => void;
  applyEvent: (event: WsEvent) => void;
  setSavedPoiIds: (ids: string[]) => void;
};

function upsertMessage(messages: ChatMessage[], next: ChatMessage) {
  const existing = messages.findIndex((message) => message.id === next.id);
  if (existing === -1) {
    return [...messages, next];
  }

  return messages.map((message, index) =>
    index === existing ? next : message
  );
}

export const useSessionStore = create<SessionState>((set) => ({
  snapshot: undefined,
  streamingMessage: undefined,
  activeTurnId: undefined,
  isStreaming: false,
  error: undefined,
  reset: () =>
    set({
      snapshot: undefined,
      streamingMessage: undefined,
      activeTurnId: undefined,
      isStreaming: false,
      error: undefined
    }),
  hydrate: (snapshot) =>
    set({
      snapshot,
      streamingMessage: undefined,
      activeTurnId: undefined,
      isStreaming: false,
      error: undefined
    }),
  setSavedPoiIds: (ids) =>
    set((state) => ({
      snapshot: state.snapshot
        ? {
            ...state.snapshot,
            savedPoiIds: ids
          }
        : state.snapshot
    })),
  applyEvent: (event) =>
    set((state) => {
      switch (event.type) {
        case "session.snapshot":
          return {
            snapshot: event.data,
            streamingMessage:
              state.streamingMessage &&
              event.data.messages.some(
                (message) => message.id === state.streamingMessage?.id
              )
                ? undefined
                : state.streamingMessage,
            error: undefined
          };
        case "turn.started":
          return {
            activeTurnId: event.data.turnId,
            isStreaming: true,
            error: undefined
          };
        case "message.delta": {
          const base =
            state.streamingMessage?.id === event.data.messageId
              ? state.streamingMessage
              : {
                  id: event.data.messageId,
                  sessionId: event.data.sessionId,
                  role: event.data.role,
                  content: "",
                  createdAt: new Date().toISOString(),
                  phase: "draft" as const,
                  meta: { turnId: event.data.turnId }
                };

          return {
            streamingMessage: {
              ...base,
              content: `${base.content}${event.data.delta}`
            },
            isStreaming: !event.data.done
          };
        }
        case "message.committed":
          return {
            snapshot: state.snapshot
              ? {
                  ...state.snapshot,
                  messages: upsertMessage(state.snapshot.messages, event.data)
                }
              : state.snapshot,
            streamingMessage:
              state.streamingMessage?.id === event.data.id
                ? undefined
                : state.streamingMessage
          };
        case "plan.updated":
          return {
            snapshot: state.snapshot
              ? {
                  ...state.snapshot,
                  plan: event.data.plan,
                  poiCatalog: event.data.poiCatalog,
                  title: event.data.plan.title
                }
              : state.snapshot
          };
        case "turn.completed":
          return {
            activeTurnId: undefined,
            isStreaming: false,
            error: undefined
          };
        case "turn.failed":
          return {
            activeTurnId: undefined,
            isStreaming: false,
            error: event.data.error
          };
        case "trace.updated":
          return {
            snapshot: state.snapshot
              ? {
                  ...state.snapshot,
                  traces: [...state.snapshot.traces, event.data]
                }
              : state.snapshot
          };
        default:
          return state;
      }
    })
}));

export function getVisibleMessages(
  snapshot?: CanonicalSession,
  streamingMessage?: ChatMessage
) {
  if (!snapshot && !streamingMessage) {
    return [];
  }

  const messages = [...(snapshot?.messages || [])];
  if (
    streamingMessage &&
    !messages.some((message) => message.id === streamingMessage.id)
  ) {
    messages.push(streamingMessage);
  }

  return messages.sort((left, right) =>
    left.createdAt.localeCompare(right.createdAt)
  );
}
