import { z } from "zod";

const EnvSchema = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PORT: z.string().optional(),
  GEMINI_API_KEY: z.string().optional(),
  GEMINI_MODEL_FAST: z.string().default("gemini-2.5-flash"),
  GEMINI_MODEL_BALANCED: z.string().default("gemini-2.5-flash"),
  GEMINI_MODEL_DEEP: z.string().default("gemini-2.5-pro"),
  GEMINI_MODEL_ROUTER: z.string().default("gemma-4-31b-it"),
  GEMINI_MODEL_ROUTER_FALLBACK: z.string().default("gemini-2.5-flash"),
  GEMINI_MODEL_NARRATIVE: z.string().default("gemini-flash-latest"),
  GEMINI_MODEL_NARRATIVE_FALLBACK: z.string().default("gemini-2.5-flash"),
  GEMINI_MODEL_GROUNDING: z.string().default("gemini-2.5-flash"),
  GEMINI_MODEL_GROUNDING_FALLBACK: z.string().default("gemini-2.5-flash-lite"),
  GEMINI_MODEL_EMBEDDING: z.string().default("gemini-embedding-001"),
  OPENAI_API_KEY: z.string().optional(),
  OPENAI_MODEL_FAST: z.string().default("gpt-5.4-mini"),
  OPENAI_MODEL_BALANCED: z.string().default("gpt-5.4-mini"),
  OPENAI_MODEL_DEEP: z.string().default("gpt-5.4"),
  GOOGLE_MAPS_API_KEY: z.string().optional(),
  TAVILY_API_KEY: z.string().optional(),
  SUPABASE_URL: z.string().optional(),
  SUPABASE_ANON_KEY: z.string().optional(),
  SUPABASE_SERVICE_ROLE_KEY: z.string().optional(),
  ROAMEO_ENCRYPTION_SECRET: z.string().optional(),
  APP_BASE_URL: z.string().optional(),
  WS_BASE_URL: z.string().optional(),
});

export type Env = z.infer<typeof EnvSchema>;

export const env: Env = EnvSchema.parse(process.env);

export const config = {
  port: Number(env.PORT || 4000),
  models: {
    gemini: {
      fast: env.GEMINI_MODEL_FAST,
      balanced: env.GEMINI_MODEL_BALANCED,
      deep: env.GEMINI_MODEL_DEEP
    },
    geminiTasks: {
      router: [env.GEMINI_MODEL_ROUTER, env.GEMINI_MODEL_ROUTER_FALLBACK],
      narrative: [env.GEMINI_MODEL_NARRATIVE, env.GEMINI_MODEL_NARRATIVE_FALLBACK],
      grounding: [env.GEMINI_MODEL_GROUNDING, env.GEMINI_MODEL_GROUNDING_FALLBACK],
      embedding: env.GEMINI_MODEL_EMBEDDING
    },
    openai: {
      fast: env.OPENAI_MODEL_FAST,
      balanced: env.OPENAI_MODEL_BALANCED,
      deep: env.OPENAI_MODEL_DEEP
    }
  },
};
