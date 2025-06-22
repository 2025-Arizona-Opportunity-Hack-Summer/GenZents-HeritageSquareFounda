import { supabase } from "@/lib/supabase/client"
import type { AIModel, Message, AIProvider, GenerationOptions } from "@/lib/types"

// Remove duplicate interfaces - now using centralized types

export class AIIntegration {
  private static providers: Map<string, AIProvider> = new Map()

  static registerProvider(provider: AIProvider) {
    this.providers.set(provider.name, provider)
  }

  static async generateResponse(content: string, options: GenerationOptions): Promise<string> {
    const model = await this.getModel(options.model)
    if (!model) {
      throw new Error(`Model ${options.model} not found`)
    }

    const provider = this.providers.get(model.provider)
    if (!provider) {
      throw new Error(`Provider ${model.provider} not configured`)
    }

    switch (model.provider) {
      case "openai":
        return this.generateOpenAIResponse(content, options, provider)
      case "anthropic":
        return this.generateAnthropicResponse(content, options, provider)
      case "google":
        return this.generateGoogleResponse(content, options, provider)
      case "local":
        return this.generateLocalResponse(content, options, provider)
      default:
        throw new Error(`Unsupported provider: ${model.provider}`)
    }
  }

  private static async generateOpenAIResponse(
    content: string,
    options: GenerationOptions,
    provider: AIProvider
  ): Promise<string> {
    const messages = [
      ...(options.systemPrompt ? [{ role: "system", content: options.systemPrompt }] : []),
      ...(options.context || []).map((msg) => ({
        role: msg.role,
        content: msg.content,
      })),
      { role: "user", content },
    ]

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${provider.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: options.model,
        messages,
        temperature: options.temperature || 0.7,
        max_tokens: options.maxTokens || 2048,
        top_p: options.topP || 1,
        frequency_penalty: options.frequencyPenalty || 0,
        presence_penalty: options.presencePenalty || 0,
        stop: options.stop,
        stream: options.stream || false,
      }),
    })

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`)
    }

    const data = await response.json()
    return data.choices[0].message.content
  }

  private static async generateAnthropicResponse(
    content: string,
    options: GenerationOptions,
    provider: AIProvider
  ): Promise<string> {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": provider.apiKey,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: options.model,
        max_tokens: options.maxTokens || 2048,
        temperature: options.temperature || 0.7,
        system: options.systemPrompt,
        messages: [
          ...(options.context || []).map((msg) => ({
            role: msg.role === "assistant" ? "assistant" : "user",
            content: msg.content,
          })),
          { role: "user", content },
        ],
      }),
    })

    if (!response.ok) {
      throw new Error(`Anthropic API error: ${response.statusText}`)
    }

    const data = await response.json()
    return data.content[0].text
  }

  private static async generateGoogleResponse(
    content: string,
    options: GenerationOptions,
    provider: AIProvider
  ): Promise<string> {
    // Implementation for Google AI (Gemini)
    throw new Error("Google AI integration not implemented")
  }

  private static async generateLocalResponse(
    content: string,
    options: GenerationOptions,
    provider: AIProvider
  ): Promise<string> {
    // Implementation for local models (Ollama, etc.)
    const response = await fetch(`${provider.baseUrl}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: options.model,
        prompt: content,
        temperature: options.temperature || 0.7,
        max_tokens: options.maxTokens || 2048,
        stream: false,
      }),
    })

    if (!response.ok) {
      throw new Error(`Local AI error: ${response.statusText}`)
    }

    const data = await response.json()
    return data.response
  }

  static async getModel(modelId: string): Promise<AIModel | null> {
    const { data, error } = await supabase.from("ai_models").select("*").eq("id", modelId).single()

    if (error) return null
    return data
  }

  static async getAvailableModels(): Promise<AIModel[]> {
    const { data, error } = await supabase.from("ai_models").select("*").eq("status", "active").order("name")

    if (error) return []
    return data || []
  }

  static async createCustomModel(model: Partial<AIModel>): Promise<string> {
    const { data, error } = await supabase
      .from("ai_models")
      .insert({
        ...model,
        created_by: (await supabase.auth.getUser()).data.user?.id,
        created_at: new Date().toISOString(),
      })
      .select()
      .single()

    if (error) throw error
    return data.id
  }

  static async updateModelSettings(modelId: string, settings: any): Promise<void> {
    const { error } = await supabase.from("ai_models").update({ settings }).eq("id", modelId)

    if (error) throw error
  }

  static async analyzeContent(content: string): Promise<{
    sentiment: "positive" | "negative" | "neutral"
    topics: string[]
    entities: any[]
    language: string
    confidence: number
  }> {
    // Implementation for content analysis
    // This could use various AI services for sentiment analysis, NER, etc.
    return {
      sentiment: "neutral",
      topics: [],
      entities: [],
      language: "en",
      confidence: 0.8,
    }
  }

  static async generateSummary(messages: Message[]): Promise<string> {
    const content = messages.map((m) => `${m.role}: ${m.content}`).join("\n")

    return this.generateResponse(`Please provide a concise summary of the following conversation:\n\n${content}`, {
      model: "gpt-3.5-turbo",
      temperature: 0.3,
      maxTokens: 500,
      systemPrompt: "You are a helpful assistant that creates concise summaries of conversations.",
    })
  }

  static async generateTitle(messages: Message[]): Promise<string> {
    const content = messages
      .slice(0, 5)
      .map((m) => m.content)
      .join(" ")

    return this.generateResponse(
      `Generate a short, descriptive title for a conversation that starts with: "${content.substring(0, 200)}..."`,
      {
        model: "gpt-3.5-turbo",
        temperature: 0.5,
        maxTokens: 50,
        systemPrompt:
          "Generate short, descriptive titles for conversations. Return only the title, no quotes or extra text.",
      }
    )
  }
}
