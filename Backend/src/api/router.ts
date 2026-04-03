import { randomUUID } from "node:crypto";
import { Router, type NextFunction, type Response } from "express";
import {
  createSessionInputSchema,
  planMutationInputSchema,
  providerCredentialInputSchema,
  providerSchema,
  sendMessageInputSchema,
  sessionMutationSchema,
  streamEventSchema,
  type StreamEvent,
  userSettingsUpdateSchema
} from "@roameo/contracts";
import { z } from "zod";
import { env } from "../config/env.js";
import { encryptSecret } from "../core/encryption.js";
import { StreamHub } from "../core/stream-hub.js";
import {
  authenticateUser,
  optionalAuth,
  type AuthenticatedRequest
} from "../middleware/auth.js";
import { TurnRunner } from "../runtime/turn-runner.js";
import { SessionRepository } from "../services/session-repository.js";
import { PlanMutationService } from "../services/plan-mutation-service.js";
import { DestinationImageService } from "../tools/destination-images.js";

type RouterDeps = {
  repository: SessionRepository;
  planMutationService: PlanMutationService;
  turnRunner: TurnRunner;
  streamHub: StreamHub;
};

const savePoiBodySchema = z.object({
  poiId: z.string().min(1),
  saved: z.boolean()
});

function sendEvent(response: Response, event: StreamEvent) {
  const payload = streamEventSchema.parse(event);
  response.write(`event: ${payload.type}\n`);
  response.write(`data: ${JSON.stringify(payload)}\n\n`);
}

export function buildApiRouter({
  planMutationService,
  repository,
  turnRunner,
  streamHub
}: RouterDeps) {
  const router = Router();

  router.get("/health", (_req, res) => {
    res.status(200).json({
      status: "ok",
      timestamp: new Date().toISOString()
    });
  });

  router.get("/maps/api-key", (_req, res) => {
    if (!env.GOOGLE_MAPS_API_KEY) {
      return res
        .status(503)
        .json({ error: "Google Maps API key is not configured" });
    }

    return res.json({ apiKey: env.GOOGLE_MAPS_API_KEY });
  });

  router.get("/maps/reverse-geocode", async (req, res) => {
    if (!env.GOOGLE_MAPS_API_KEY) {
      return res.status(503).json({ error: "Google Maps is not configured" });
    }

    const lat = Number.parseFloat(String(req.query.lat || ""));
    const lng = Number.parseFloat(String(req.query.lng || ""));
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
      return res.status(400).json({ error: "lat and lng query parameters are required" });
    }

    const geocodeUrl =
      "https://maps.googleapis.com/maps/api/geocode/json" +
      `?latlng=${encodeURIComponent(`${lat},${lng}`)}` +
      `&key=${encodeURIComponent(env.GOOGLE_MAPS_API_KEY)}`;

    try {
      const response = await fetch(geocodeUrl, {
        signal: AbortSignal.timeout(10_000),
        headers: { "User-Agent": "Roameo/2.0" }
      });

      if (!response.ok) {
        return res
          .status(502)
          .json({ error: `Google Geocoding error: ${response.status}` });
      }

      const payload = (await response.json()) as {
        status?: string;
        results?: Array<{
          formatted_address?: string;
          address_components?: Array<{
            long_name?: string;
            short_name?: string;
            types?: string[];
          }>;
        }>;
      };

      const result = payload.results?.[0];
      if (!result) {
        return res.status(404).json({ error: "No location found for these coordinates" });
      }

      const readComponent = (...types: string[]) =>
        result.address_components?.find((component) =>
          types.some((type) => component.types?.includes(type))
        )?.long_name;

      const locality =
        readComponent("locality", "postal_town", "administrative_area_level_3") ||
        readComponent("sublocality", "sublocality_level_1");
      const region = readComponent("administrative_area_level_1");
      const country = readComponent("country");
      const label = [locality, region].filter(Boolean).join(", ") || result.formatted_address;

      return res.json({
        label,
        locality,
        region,
        country,
        formatted: result.formatted_address || label
      });
    } catch (error) {
      return res.status(502).json({
        error:
          error instanceof Error
            ? error.message
            : "Failed to reverse geocode the current location"
      });
    }
  });

  router.get("/destination-image", async (req, res) => {
    const destination = String(req.query.q || "");
    if (!destination) {
      return res.status(400).json({ error: "q query parameter is required" });
    }

    try {
      const destinationImageService = new DestinationImageService();
      const result = await destinationImageService.getDestinationImage(destination);
      if (result.imageUrl) {
        return res.json({ imageUrl: result.imageUrl });
      }
      return res.status(404).json({ error: "Image not found" });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Destination image fetch failed";
      return res.status(502).json({ error: message });
    }
  });

  router.get("/proxy/photo", async (req, res) => {
    if (!env.GOOGLE_MAPS_API_KEY) {
      return res.status(503).json({ error: "Google Maps is not configured" });
    }

    const photoReference = String(req.query.photo_reference || "");
    const maxwidth = String(req.query.maxwidth || "900");
    if (!photoReference) {
      return res
        .status(400)
        .json({ error: "photo_reference query parameter is required" });
    }

    const photoUrl =
      "https://maps.googleapis.com/maps/api/place/photo" +
      `?photo_reference=${encodeURIComponent(photoReference)}` +
      `&maxwidth=${encodeURIComponent(maxwidth)}` +
      `&key=${encodeURIComponent(env.GOOGLE_MAPS_API_KEY)}`;

    try {
      const response = await fetch(photoUrl, {
        signal: AbortSignal.timeout(10_000),
        headers: { "User-Agent": "Roameo/2.0" }
      });

      if (!response.ok) {
        return res
          .status(response.status)
          .json({ error: `Photo service failed with ${response.status}` });
      }

      const image = Buffer.from(await response.arrayBuffer());
      res.set({
        "Content-Type": response.headers.get("content-type") || "image/jpeg",
        "Cache-Control": "public, max-age=86400",
        "Access-Control-Allow-Origin": "*"
      });
      return res.send(image);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Photo proxy failed";
      return res.status(504).json({ error: message });
    }
  });

  router.use(optionalAuth);

  router.get(
    "/sessions",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const sessions = await repository.listSessions(req.userId!);
      return res.json({ sessions });
    }
  );

  router.post(
    "/sessions",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const input = createSessionInputSchema.parse(req.body ?? {});
      const session = await repository.createSession(req.userId, input);

      if (input.initialMessage) {
        const stored =
          (await repository.getSession(session.id, req.userId)) || session;
        void turnRunner
          .runTurn(stored, req.userId, {
            content: input.initialMessage,
            providerSettings: input.providerSettings
          })
          .catch((error) => {
            console.error("Background turn failed:", error);
          });
      }

      return res.status(201).json(session);
    }
  );

  router.get(
    "/sessions/:sessionId",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const sessionId = String(req.params.sessionId);
      const session = await repository.getSession(
        sessionId,
        req.userId
      );
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }
      return res.json(session);
    }
  );

  router.patch(
    "/sessions/:sessionId",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const sessionId = String(req.params.sessionId);
      const mutation = sessionMutationSchema.parse(req.body ?? {});
      const session = await repository.getSession(
        sessionId,
        req.userId
      );
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      const updated = await repository.updateSession(session.id, mutation);
      if (!updated) {
        return res.status(404).json({ error: "Session not found" });
      }

      streamHub.emit(updated.id, {
        type: "session.snapshot",
        data: updated
      });

      return res.json(updated);
    }
  );

  router.delete(
    "/sessions/:sessionId",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      await repository.deleteSession(String(req.params.sessionId), req.userId);
      return res.status(204).send();
    }
  );

  router.post(
    "/sessions/:sessionId/plan-mutations",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const sessionId = String(req.params.sessionId);
      const mutation = planMutationInputSchema.parse(req.body ?? {});
      const session = await repository.getSession(sessionId, req.userId);
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      const updated = await planMutationService.apply(
        session,
        req.userId,
        mutation
      );

      if (updated.plan) {
        streamHub.emit(updated.id, {
          type: "plan.updated",
          data: {
            sessionId: updated.id,
            plan: updated.plan,
            poiCatalog: updated.poiCatalog
          }
        });
      }
      streamHub.emit(updated.id, {
        type: "session.snapshot",
        data: updated
      });

      return res.json(updated);
    }
  );

  router.post(
    "/sessions/:sessionId/messages",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const sessionId = String(req.params.sessionId);
      const input = sendMessageInputSchema.parse(req.body ?? {});
      const session = await repository.getSession(
        sessionId,
        req.userId
      );
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      void turnRunner.runTurn(session, req.userId, input).catch((error) => {
        console.error("Turn failed:", error);
      });

      return res.status(202).json({
        accepted: true,
        sessionId: session.id
      });
    }
  );

  router.get(
    "/sessions/:sessionId/stream",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const sessionId = String(req.params.sessionId);
      const session = await repository.getSession(
        sessionId,
        req.userId
      );
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders?.();

      const close = streamHub.attach(session.id, randomUUID(), res);
      sendEvent(res, {
        type: "session.snapshot",
        data: session
      });

      const keepAlive = setInterval(() => {
        res.write(": keepalive\n\n");
      }, 15_000);

      req.on("close", () => {
        clearInterval(keepAlive);
        close();
      });
    }
  );

  router.get(
    "/sessions/:sessionId/saved-pois",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const sessionId = String(req.params.sessionId);
      const session = await repository.getSession(
        sessionId,
        req.userId
      );
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      return res.json({ ids: session.savedPoiIds });
    }
  );

  router.post(
    "/sessions/:sessionId/saved-pois",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const sessionId = String(req.params.sessionId);
      const body = savePoiBodySchema.parse(req.body ?? {});
      const session = await repository.getSession(
        sessionId,
        req.userId
      );
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      const updated = await repository.setSavedPoi(
        session.id,
        body.poiId,
        body.saved
      );
      if (!updated) {
        return res.status(404).json({ error: "Session not found" });
      }

      streamHub.emit(updated.id, {
        type: "session.snapshot",
        data: updated
      });

      return res.json({ ids: updated.savedPoiIds });
    }
  );

  router.get("/me/settings", authenticateUser, async (req: AuthenticatedRequest, res) => {
    const settings = await repository.getUserSettings(req.userId!);
    return res.json({
      providerSettings: settings.providerSettings,
      preferences: settings.preferences,
      credentials: providerSchema.options.map((provider: "gemini" | "openai") => ({
        provider,
        keySource: "user",
        configured: Boolean(settings.credentials[provider]?.encryptedKey),
        lastUpdatedAt: settings.credentials[provider]?.updatedAt
      }))
    });
  });

  router.put("/me/settings", authenticateUser, async (req: AuthenticatedRequest, res) => {
    const body = userSettingsUpdateSchema.parse(req.body ?? {});
    const settings = await repository.saveUserSettings(
      req.userId!,
      body.providerSettings,
      body.preferences
    );

    return res.json({
      providerSettings: settings.providerSettings,
      preferences: settings.preferences
    });
  });

  router.put(
    "/me/credentials/:provider",
    authenticateUser,
    async (req: AuthenticatedRequest, res) => {
      const provider = providerSchema.parse(req.params.provider);
      const body = providerCredentialInputSchema.parse(req.body ?? {});

      await repository.saveUserCredential(
        req.userId!,
        provider,
        "user",
        encryptSecret(body.apiKey)
      );

      return res.status(204).send();
    }
  );

  router.use((
    error: unknown,
    _req: AuthenticatedRequest,
    res: Response,
    _next: NextFunction
  ) => {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Invalid request",
        details: error.flatten()
      });
    }

    const message =
      error instanceof Error ? error.message : "Unexpected server error";
    console.error(message, error);
    return res.status(500).json({ error: message });
  });

  return router;
}
