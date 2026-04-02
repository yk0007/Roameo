import { BACKEND_URL, type WsEvent } from "./types";

type StreamCallbacks = {
  getAccessToken?: () => Promise<string | null>;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Error) => void;
};

function parseChunk(buffer: string, onEvent: (event: WsEvent) => void) {
  const segments = buffer.split("\n\n");
  const remainder = segments.pop() || "";

  for (const segment of segments) {
    const lines = segment.split("\n");
    const dataLine = lines.find((line) => line.startsWith("data: "));
    if (!dataLine) {
      continue;
    }

    const payload = dataLine.slice(6);
    try {
      onEvent(JSON.parse(payload) as WsEvent);
    } catch (error) {
      console.error("Failed to parse stream payload", error);
    }
  }

  return remainder;
}

export function connectWs(
  sessionId: string,
  onEvent: (event: WsEvent) => void,
  callbacks: StreamCallbacks = {}
) {
  const controller = new AbortController();
  const decoder = new TextDecoder();
  let closed = false;

  const connect = async () => {
    const token = await callbacks.getAccessToken?.();
    const headers = new Headers({
      Accept: "text/event-stream"
    });

    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }

    const response = await fetch(
      `${BACKEND_URL}/api/sessions/${encodeURIComponent(sessionId)}/stream`,
      {
        method: "GET",
        headers,
        credentials: "include",
        signal: controller.signal
      }
    );

    if (!response.ok || !response.body) {
      throw new Error(`Stream connection failed with ${response.status}`);
    }

    callbacks.onOpen?.();

    const reader = response.body.getReader();
    let buffer = "";

    while (!closed) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      buffer = parseChunk(buffer, onEvent);
    }
  };

  void connect()
    .catch((error) => {
      if (!closed) {
        callbacks.onError?.(
          error instanceof Error ? error : new Error("Stream failed")
        );
      }
    })
    .finally(() => {
      if (!closed) {
        callbacks.onClose?.();
      }
    });

  return {
    close() {
      closed = true;
      controller.abort();
      callbacks.onClose?.();
    }
  };
}
