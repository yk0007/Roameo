import { z } from "zod";

const EnvSchema = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PORT: z.string().optional(),
  GEMINI_API_KEY: z.string().optional(),
  GEMINI_MODEL_FAST: z.string().default("gemini-2.5-flash"),
  GEMINI_MODEL_BALANCED: z.string().default("gemini-2.5-flash"),
  GEMINI_MODEL_DEEP: z.string().default("gemini-2.5-pro"),
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
    openai: {
      fast: env.OPENAI_MODEL_FAST,
      balanced: env.OPENAI_MODEL_BALANCED,
      deep: env.OPENAI_MODEL_DEEP
    }
  },
};
