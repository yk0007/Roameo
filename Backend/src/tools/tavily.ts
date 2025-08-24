import { env } from "../config/env.js";

export interface TavilyQuery { q: string }
export interface TavilyResult { title: string; url: string; snippet?: string }

export class TavilyClient {
  async search(_q: TavilyQuery): Promise<TavilyResult[]> {
    if (!env.TAVILY_API_KEY) return [];
    // TODO: call Tavily API
    return [];
  }
}
