import { z } from "zod";

const EnvSchema = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PORT: z.string().optional(),
  GEMINI_API_KEY: z.string().optional(),
  GEMINI_MODEL_FLASH: z.string().default("gemini-2.5-flash"),
  GEMINI_MODEL_PRO: z.string().default("gemini-2.5-pro"),
  GOOGLE_MAPS_API_KEY: z.string().optional(),
  TAVILY_API_KEY: z.string().optional(),
  SUPABASE_URL: z.string().optional(),
  SUPABASE_ANON_KEY: z.string().optional(),
  SUPABASE_SERVICE_ROLE_KEY: z.string().optional(),
  APP_BASE_URL: z.string().optional(),
  WS_BASE_URL: z.string().optional(),
});

export type Env = z.infer<typeof EnvSchema>;

export const env: Env = EnvSchema.parse(process.env);

export const config = {
  port: Number(env.PORT || 4000),
  models: {
    flash: env.GEMINI_MODEL_FLASH,
    pro: env.GEMINI_MODEL_PRO,
  },
};
