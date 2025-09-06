import Groq from "groq-sdk";

export type GroqModel =
  | "llama-3.1-8b-instant"
  | "llama-3.3-70b-versatile";

export class GroqClient {
  private client: Groq;
  private model: GroqModel;

  constructor(opts: { model?: GroqModel } = {}) {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      throw new Error(
        "Missing GROQ_API_KEY. Set it in your environment before starting the server.",
      );
    }
    this.client = new Groq({ apiKey });
    this.model = opts.model || "llama-3.1-8b-instant"; // efficient default
  }

  async chat(
    prompt: string,
    options?: { temperature?: number; system?: string },
  ): Promise<string> {
    const completion = await this.client.chat.completions.create({
      model: this.model,
      temperature: options?.temperature ?? 0.6,
      messages: [
        {
          role: "system",
          content:
            options?.system ||
            "You are a helpful, precise assistant. Keep responses well-structured and on task.",
        },
        { role: "user", content: prompt },
      ],
    });

    const content = completion.choices?.[0]?.message?.content || "";
    return content;
  }
}
