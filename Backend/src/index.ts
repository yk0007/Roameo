import "dotenv/config";
import cors from "cors";
import express from "express";
import { buildApiRouter } from "./api/router.js";
import { config } from "./config/env.js";
import { StreamHub } from "./core/stream-hub.js";
import { TurnRunner } from "./runtime/turn-runner.js";
import { ProviderService } from "./services/provider-service.js";
import { PlanMutationService } from "./services/plan-mutation-service.js";
import { SessionRepository } from "./services/session-repository.js";
import { TravelToolsService } from "./services/travel-tools.js";

const app = express();
const streamHub = new StreamHub();
const repository = new SessionRepository();
const providerService = new ProviderService(repository);
const travelTools = new TravelToolsService();
const planMutationService = new PlanMutationService(
  repository,
  providerService,
  travelTools
);
const turnRunner = new TurnRunner(
  repository,
  providerService,
  travelTools,
  streamHub
);

app.disable("x-powered-by");
app.use(
  cors({
    origin: [
      "http://localhost:3000",
      "https://roameo-app.vercel.app",
      "https://roameo.onrender.com",
      /^https:\/\/.*\.vercel\.app$/,
      /^https:\/\/.*\.onrender\.com$/
    ],
    credentials: true,
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With"]
  })
);
app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req, res) => {
  res.status(200).json({
    status: "ok",
    service: "roameo-backend",
    timestamp: new Date().toISOString()
  });
});

app.use(
  "/api",
  buildApiRouter({
    planMutationService,
    repository,
    travelTools,
    turnRunner,
    streamHub
  })
);

const host = "0.0.0.0";

app.listen(config.port, host, () => {
  console.log(`[roameo-backend] listening on ${host}:${config.port}`);
});
