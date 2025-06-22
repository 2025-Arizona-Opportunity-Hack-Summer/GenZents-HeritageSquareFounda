import { createClient } from "@/lib/supabase/client"
import { logger } from "@/lib/utils/logger"

export interface ChatMessage {
  role: "user" | "assistant" | "system"
  content: string
  chatId: string
}

export interface StreamingResponse {
  type: "chunk" | "complete" | "error" | "message" | "done"
  data?: any
  content?: string
  message?: any
  error?: string
}

export class AIService {
  private static baseURL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

  static async getAuthToken(): Promise<string | null> {
    const supabase = createClient()
    const { data } = await supabase.auth.getSession()
    return data.session?.access_token || null
  }

  static async sendMessage(chatId: string, content: string): Promise<any> {
    const token = await this.getAuthToken()

    if (!token) {
      logger.error("No auth token available", undefined, { component: "ai-service" })
      throw new Error("User not authenticated")
    }

    const url = `${this.baseURL}/api/messages/chat/${chatId}`
    logger.info("Sending message to backend", { component: "ai-service", url, chatId, hasContent: !!content })

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          role: "user",
          content,
        }),
      })

      logger.info("Backend response received", { component: "ai-service", status: response.status, ok: response.ok })

      if (!response.ok) {
        let errorDetail = "Failed to send message"
        try {
          const error = await response.json()
          errorDetail = error.detail || errorDetail
          logger.error("Backend error response", undefined, { component: "ai-service", error, status: response.status })
        } catch {
          errorDetail = `HTTP ${response.status}: ${response.statusText}`
          logger.error("Failed to parse error response", undefined, { component: "ai-service", status: response.status, statusText: response.statusText })
        }
        
        if (response.status === 404) {
          throw new Error("Chat not found")
        }
        
        logger.error("Send message failed", new Error(errorDetail), { component: "ai-service", status: response.status })
        throw new Error(errorDetail)
      }

      const result = await response.json()
      logger.info("Message sent successfully", { component: "ai-service", messageId: result.id })
      return result
    } catch (error) {
      logger.error("Network error while sending message", error as Error, { component: "ai-service", url })
      throw error
    }
  }

  static async createChat(title: string = "New Chat", systemPrompt?: string): Promise<string> {
    const token = await this.getAuthToken()

    if (!token) {
      logger.error("No auth token available for chat creation", undefined, { component: "ai-service" })
      throw new Error("User not authenticated")
    }

    const url = `${this.baseURL}/api/chats/`
    logger.info("Creating chat on backend", { component: "ai-service", url, title, hasSystemPrompt: !!systemPrompt })

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          title,
          system_prompt: systemPrompt,
        }),
      })

      logger.info("Chat creation response received", { component: "ai-service", status: response.status, ok: response.ok })

      if (!response.ok) {
        let errorDetail = "Failed to create chat"
        try {
          const error = await response.json()
          errorDetail = error.detail || errorDetail
          logger.error("Backend chat creation error", undefined, { component: "ai-service", error, status: response.status })
        } catch {
          errorDetail = `HTTP ${response.status}: ${response.statusText}`
          logger.error("Failed to parse chat creation error", undefined, { component: "ai-service", status: response.status, statusText: response.statusText })
        }
        
        logger.error("Create chat failed", new Error(errorDetail), { component: "ai-service", status: response.status })
        throw new Error(errorDetail)
      }

      const chat = await response.json()
      logger.info("Chat created successfully", { component: "ai-service", chatId: chat.id, title: chat.title })
      return chat.id
    } catch (error) {
      logger.error("Network error while creating chat", error as Error, { component: "ai-service", url })
      throw error
    }
  }

  static async sendMessageStream(chatId: string, content: string): Promise<ReadableStream> {
    const token = await this.getAuthToken()

    if (!token) {
      logger.error("No auth token available for streaming", undefined, { component: "ai-service" })
      throw new Error("User not authenticated")
    }

    const url = `${this.baseURL}/api/messages/chat/${chatId}/stream`
    logger.info("Starting message stream to backend", { component: "ai-service", url, chatId, hasContent: !!content })

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          role: "user",
          content,
        }),
      })

      logger.info("Stream response received", { component: "ai-service", status: response.status, ok: response.ok })

      if (!response.ok) {
        let errorDetail = "Failed to send message"
        try {
          const error = await response.json()
          errorDetail = error.detail || errorDetail
          logger.error("Backend stream error response", undefined, { component: "ai-service", error, status: response.status })
        } catch {
          errorDetail = `HTTP ${response.status}: ${response.statusText}`
          logger.error("Failed to parse stream error response", undefined, { component: "ai-service", status: response.status, statusText: response.statusText })
        }
        
        if (response.status === 404) {
          throw new Error("Chat not found")
        }
        
        logger.error("Send message stream failed", new Error(errorDetail), { component: "ai-service", status: response.status })
        throw new Error(errorDetail)
      }

      logger.info("Stream established successfully", { component: "ai-service" })
      return response.body!
    } catch (error) {
      logger.error("Network error while starting stream", error as Error, { component: "ai-service", url })
      throw error
    }
  }

  static async *parseStreamingResponse(stream: ReadableStream): AsyncGenerator<StreamingResponse> {
    const reader = stream.getReader()
    const decoder = new TextDecoder()

    logger.info("Starting to parse streaming response", { component: "ai-service" })

    try {
      while (true) {
        const { done, value } = await reader.read()

        if (done) {
          logger.info("Stream reading completed", { component: "ai-service" })
          break
        }

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split("\n")

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              logger.debug("Parsed streaming data", { component: "ai-service", type: data.type })
              yield data as StreamingResponse
            } catch (e) {
              logger.warn("Failed to parse streaming data", { component: "ai-service", line, error: e })
            }
          }
        }
      }
    } catch (error) {
      logger.error("Error reading stream", error as Error, { component: "ai-service" })
      throw error
    } finally {
      reader.releaseLock()
      logger.info("Stream reader released", { component: "ai-service" })
    }
  }

  static async getChatMessages(chatId: string, limit = 50, offset = 0): Promise<any[]> {
    const token = await this.getAuthToken()

    if (!token) {
      throw new Error("User not authenticated")
    }

    const response = await fetch(`${this.baseURL}/api/messages/chat/${chatId}?limit=${limit}&offset=${offset}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Failed to fetch messages")
    }

    return response.json()
  }

  static async updateMessage(messageId: string, content: string): Promise<any> {
    const token = await this.getAuthToken()

    if (!token) {
      throw new Error("User not authenticated")
    }

    const response = await fetch(`${this.baseURL}/api/messages/${messageId}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        content,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Failed to update message")
    }

    return response.json()
  }

  static async deleteMessage(messageId: string): Promise<void> {
    const token = await this.getAuthToken()

    if (!token) {
      throw new Error("User not authenticated")
    }

    const response = await fetch(`${this.baseURL}/api/messages/${messageId}`, {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Failed to delete message")
    }
  }
}
