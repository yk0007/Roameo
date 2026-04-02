import type { Response } from "express";
import type { StreamEvent } from "@roameo/contracts";

type Client = {
  id: string;
  response: Response;
};

export class StreamHub {
  private sessions = new Map<string, Map<string, Client>>();

  attach(sessionId: string, clientId: string, response: Response): () => void {
    if (!this.sessions.has(sessionId)) {
      this.sessions.set(sessionId, new Map());
    }

    this.sessions.get(sessionId)!.set(clientId, {
      id: clientId,
      response
    });

    response.write(": connected\n\n");

    return () => {
      const clients = this.sessions.get(sessionId);
      if (!clients) {
        return;
      }

      clients.delete(clientId);
      if (clients.size === 0) {
        this.sessions.delete(sessionId);
      }
    };
  }

  emit(sessionId: string, event: StreamEvent): void {
    const clients = this.sessions.get(sessionId);
    if (!clients?.size) {
      return;
    }

    const payload = `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;

    for (const client of clients.values()) {
      client.response.write(payload);
    }
  }
}
