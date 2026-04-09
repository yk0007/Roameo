import { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";
import type {
  KeySource,
  Provider,
  RunMode,
  SessionProviderSettings
} from "@roameo/contracts";
import { z, type ZodSchema } from "zod";
import { config, env } from "../config/env.js";
import { decryptSecret } from "../core/encryption.js";
import { SessionRepository } from "./session-repository.js";

export type ResolvedProvider = {
  provider: Provider;
  keySource: KeySource;
  model: string;
  apiKey: string;
};

type GenerateTextArgs = {
  resolved: ResolvedProvider;
  instructions: string;
  input: string;
  models?: string[];
};

type GenerateObjectArgs<T> = GenerateTextArgs & {
  schema: ZodSchema<T>;
  schemaName: string;
};

function pickModel(provider: Provider, runMode: RunMode): string {
  if (provider === "openai") {
    return config.models.openai[runMode];
  }
  return config.models.gemini[runMode];
}

function parseJsonBlock<T>(text: string, schema: ZodSchema<T>): T {
  const match = text.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
  if (!match) {
    throw new Error("Model response did not contain JSON");
  }

  return schema.parse(JSON.parse(match[0]));
}

function candidateModels(resolved: ResolvedProvider, overrides?: string[]): string[] {
  const models = [...(overrides || []), resolved.model].filter(Boolean);
  return Array.from(new Set(models));
}

async function generateOpenAIText(args: GenerateTextArgs): Promise<string> {
  const client = new OpenAI({ apiKey: args.resolved.apiKey });
  const response = await client.responses.create({
    model: args.resolved.model,
    store: false,
    instructions: args.instructions,
    input: args.input
  });

  return response.output_text?.trim() || "";
}

async function generateGeminiText(args: GenerateTextArgs): Promise<string> {
  const ai = new GoogleGenAI({ apiKey: args.resolved.apiKey });
  let lastError: unknown;
  for (const model of candidateModels(args.resolved, args.models)) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: args.input,
        config: {
          systemInstruction: args.instructions,
          temperature: 0.3,
          topP: 0.9,
          maxOutputTokens: 8192
        }
      });

      return response.text?.trim() || "";
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Gemini text generation failed");
}

async function generateGeminiObject<T>(
  args: GenerateObjectArgs<T>
): Promise<T> {
  const ai = new GoogleGenAI({ apiKey: args.resolved.apiKey });
  let lastError: unknown;
  for (const model of candidateModels(args.resolved, args.models)) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: args.input,
        config: {
          systemInstruction: args.instructions,
          temperature: 0.2,
          topP: 0.9,
          maxOutputTokens: 8192,
          responseMimeType: "application/json",
          responseJsonSchema: z.toJSONSchema(args.schema)
        }
      });

      const text = response.text?.trim();
      if (!text) {
        throw new Error("Gemini returned an empty structured response");
      }

      return parseJsonBlock(text, args.schema);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Gemini structured generation failed");
}

export class ProviderService {
  constructor(private repository: SessionRepository) {}

  async resolveProvider(
    userId: string | undefined,
    sessionSettings?: SessionProviderSettings
  ): Promise<ResolvedProvider> {
    const provider = sessionSettings?.provider || "gemini";
    const keySource = sessionSettings?.keySource || "platform";
    const runMode = sessionSettings?.runMode || "balanced";
    const model = pickModel(provider, runMode);

    if (keySource === "platform") {
      const apiKey =
        provider === "openai" ? env.OPENAI_API_KEY : env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error(`Missing platform ${provider} API key`);
      }
      return { provider, keySource, model, apiKey };
    }

    if (!userId) {
      throw new Error("A signed-in user is required for BYOK");
    }

    const settings = await this.repository.getUserSettings(userId);
    const stored = settings.credentials[provider];

    if (!stored?.encryptedKey) {
      throw new Error(`No saved ${provider} API key found for this user`);
    }

    return {
      provider,
      keySource,
      model,
      apiKey: decryptSecret(stored.encryptedKey)
    };
  }

  async generateText(args: GenerateTextArgs): Promise<string> {
    if (args.resolved.provider === "openai") {
      return generateOpenAIText(args);
    }
    return generateGeminiText(args);
  }

  async generateObject<T>(args: GenerateObjectArgs<T>): Promise<T> {
    if (args.resolved.provider === "gemini") {
      return generateGeminiObject(args);
    }

    const prompt = `${args.input}

Return only valid JSON that matches the requested schema "${args.schemaName}".`;

    const text = await this.generateText({
      resolved: args.resolved,
      instructions: args.instructions,
      input: prompt
    });

    return parseJsonBlock(text, args.schema);
  }

  routerModels(resolved: ResolvedProvider): string[] | undefined {
    return resolved.provider === "gemini" ? config.models.geminiTasks.router : undefined;
  }

  narrativeModels(resolved: ResolvedProvider): string[] | undefined {
    return resolved.provider === "gemini" ? config.models.geminiTasks.narrative : undefined;
  }
}
