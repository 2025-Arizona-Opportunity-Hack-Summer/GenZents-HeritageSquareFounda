"use client"

import type React from "react"
import { createContext, useContext, useReducer, useCallback, useEffect } from "react"
import { nanoid } from "nanoid"
import { logger } from "@/lib/utils/logger"
import { SecureStorage } from "@/lib/utils/local-storage"
import type { Message, Attachment, ChatSession } from "@/lib/types/chat"

// Helper function to create a proper message
const createMessage = (
  content: string,
  role: "user" | "assistant" | "system" | "function",
  chatId: string,
  metadata?: any
): Message => ({
  id: nanoid(),
  chat_id: chatId,
  role,
  content,
  created_at: new Date().toISOString(),
  timestamp: new Date(),
  edited: false,
  deleted: false,
  metadata: {
    model: metadata?.model,
    tokens: metadata?.tokens,
    cost: metadata?.cost,
    processingTime: metadata?.processingTime,
    confidence: undefined,
    sources: [],
    citations: [],
    language: undefined,
    sentiment: undefined,
    topics: [],
    entities: [],
  },
  attachments: metadata?.attachments || [],
  reactions: [],
  mentions: [],
  status: {
    sent: role === "user",
    delivered: role !== "user",
    read: false,
    retries: 0,
    status: role === "user" ? "sending" : "delivered",
  },
  encryption: {
    encrypted: false,
  },
})

// Remove duplicate interfaces as they now come from centralized types

interface ChatState {
  sessions: ChatSession[]
  currentSessionId: string | null
  isStreaming: boolean
  isTyping: boolean
  searchQuery: string
  selectedMessages: string[]
  attachments: Attachment[]
  contextFiles: string[]
  promptHistory: string[]
}

type ChatAction =
  | { type: "CREATE_SESSION"; payload: { title?: string; systemPrompt?: string } }
  | { type: "SELECT_SESSION"; payload: string }
  | { type: "DELETE_SESSION"; payload: string }
  | { type: "UPDATE_SESSION"; payload: { id: string; updates: Partial<ChatSession> } }
  | {
      type: "ADD_MESSAGE"
      payload: {
        sessionId: string
        content: string
        role: "user" | "assistant" | "system" | "function"
        metadata?: any
      }
    }
  | { type: "UPDATE_MESSAGE"; payload: { sessionId: string; messageId: string; updates: Partial<Message> } }
  | { type: "DELETE_MESSAGE"; payload: { sessionId: string; messageId: string } }
  | { type: "SET_STREAMING"; payload: boolean }
  | { type: "SET_TYPING"; payload: boolean }
  | { type: "SET_SEARCH_QUERY"; payload: string }
  | { type: "TOGGLE_MESSAGE_SELECTION"; payload: string }
  | { type: "CLEAR_MESSAGE_SELECTION" }
  | { type: "ADD_ATTACHMENT"; payload: Attachment }
  | { type: "REMOVE_ATTACHMENT"; payload: string }
  | { type: "CLEAR_ATTACHMENTS" }
  | { type: "ADD_CONTEXT_FILE"; payload: string }
  | { type: "REMOVE_CONTEXT_FILE"; payload: string }
  | { type: "ADD_TO_PROMPT_HISTORY"; payload: string }
  | { type: "LOAD_SESSIONS"; payload: ChatSession[] }

const initialState: ChatState = {
  sessions: [],
  currentSessionId: null,
  isStreaming: false,
  isTyping: false,
  searchQuery: "",
  selectedMessages: [],
  attachments: [],
  contextFiles: [],
  promptHistory: [],
}

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "CREATE_SESSION": {
      const newSession: ChatSession = {
        id: nanoid(),
        title: action.payload.title || "New Chat",
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
        systemPrompt: action.payload.systemPrompt,
      }
      return {
        ...state,
        sessions: [newSession, ...state.sessions],
        currentSessionId: newSession.id,
      }
    }

    case "SELECT_SESSION":
      return {
        ...state,
        currentSessionId: action.payload,
        selectedMessages: [],
        attachments: [],
      }

    case "DELETE_SESSION":
      return {
        ...state,
        sessions: state.sessions.filter((s) => s.id !== action.payload),
        currentSessionId: state.currentSessionId === action.payload ? null : state.currentSessionId,
      }

    case "UPDATE_SESSION":
      return {
        ...state,
        sessions: state.sessions.map((s) =>
          s.id === action.payload.id ? { ...s, ...action.payload.updates, updatedAt: new Date() } : s
        ),
      }

    case "ADD_MESSAGE": {
      const message = createMessage(
        action.payload.content,
        action.payload.role,
        action.payload.sessionId,
        action.payload.metadata
      )

      return {
        ...state,
        sessions: state.sessions.map((s) =>
          s.id === action.payload.sessionId ? { ...s, messages: [...s.messages, message], updatedAt: new Date() } : s
        ),
      }
    }

    case "UPDATE_MESSAGE":
      return {
        ...state,
        sessions: state.sessions.map((s) =>
          s.id === action.payload.sessionId
            ? {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === action.payload.messageId ? { ...m, ...action.payload.updates } : m
                ),
                updatedAt: new Date(),
              }
            : s
        ),
      }

    case "DELETE_MESSAGE":
      return {
        ...state,
        sessions: state.sessions.map((s) =>
          s.id === action.payload.sessionId
            ? {
                ...s,
                messages: s.messages.filter((m) => m.id !== action.payload.messageId),
                updatedAt: new Date(),
              }
            : s
        ),
      }

    case "SET_STREAMING":
      return { ...state, isStreaming: action.payload }

    case "SET_TYPING":
      return { ...state, isTyping: action.payload }

    case "SET_SEARCH_QUERY":
      return { ...state, searchQuery: action.payload }

    case "TOGGLE_MESSAGE_SELECTION":
      return {
        ...state,
        selectedMessages: state.selectedMessages.includes(action.payload)
          ? state.selectedMessages.filter((id) => id !== action.payload)
          : [...state.selectedMessages, action.payload],
      }

    case "CLEAR_MESSAGE_SELECTION":
      return { ...state, selectedMessages: [] }

    case "ADD_ATTACHMENT":
      return {
        ...state,
        attachments: [...state.attachments, action.payload],
      }

    case "REMOVE_ATTACHMENT":
      return {
        ...state,
        attachments: state.attachments.filter((a) => a.id !== action.payload),
      }

    case "CLEAR_ATTACHMENTS":
      return { ...state, attachments: [] }

    case "ADD_CONTEXT_FILE":
      return {
        ...state,
        contextFiles: [...state.contextFiles, action.payload],
      }

    case "REMOVE_CONTEXT_FILE":
      return {
        ...state,
        contextFiles: state.contextFiles.filter((f) => f !== action.payload),
      }

    case "ADD_TO_PROMPT_HISTORY":
      return {
        ...state,
        promptHistory: [action.payload, ...state.promptHistory.filter((p) => p !== action.payload)].slice(0, 50),
      }

    case "LOAD_SESSIONS":
      return { ...state, sessions: action.payload }

    default:
      return state
  }
}

const ChatContext = createContext<{
  state: ChatState
  dispatch: React.Dispatch<ChatAction>
  currentSession: ChatSession | null
  sendMessage: (content: string, attachments?: Attachment[]) => Promise<void>
  regenerateMessage: (messageId: string) => Promise<void>
  stopGeneration: () => void
  exportChat: (sessionId: string, format: "json" | "markdown" | "txt") => void
  importChat: (data: any) => void
} | null>(null)

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(chatReducer, initialState)

  const currentSession = state.sessions.find((s) => s.id === state.currentSessionId) || null

  // Load sessions from localStorage on mount
  useEffect(() => {
    const loadSessions = async () => {
      try {
        const sessions = await SecureStorage.getItem<ChatSession[]>("chat-sessions")
        if (sessions) {
          const processedSessions = sessions.map((s: any) => ({
            ...s,
            createdAt: new Date(s.createdAt),
            updatedAt: new Date(s.updatedAt),
            messages: s.messages.map((m: any) => ({
              ...m,
              timestamp: new Date(m.timestamp),
            })),
          }))
          dispatch({ type: "LOAD_SESSIONS", payload: processedSessions })
        }
      } catch (error) {
        logger.error("Failed to load sessions", error as Error, { component: "chat-provider" })
      }
    }
    loadSessions()
  }, [])

  // Save sessions to localStorage when they change
  useEffect(() => {
    if (state.sessions.length > 0) {
      SecureStorage.setItem("chat-sessions", state.sessions, { 
        encrypt: true, 
        expiryHours: 24 * 7 // 7 days
      })
    }
  }, [state.sessions])

  const sendMessage = useCallback(
    async (content: string, attachments?: Attachment[]) => {
      if (!state.currentSessionId) {
        dispatch({ type: "CREATE_SESSION", payload: { title: content.slice(0, 50) + "..." } })
        return
      }

      // Add user message
      dispatch({
        type: "ADD_MESSAGE",
        payload: {
          sessionId: state.currentSessionId,
          role: "user",
          content,
          metadata: { attachments },
        },
      })

      // Add to prompt history
      dispatch({ type: "ADD_TO_PROMPT_HISTORY", payload: content })

      // Simulate AI response
      dispatch({ type: "SET_STREAMING", payload: true })
      dispatch({ type: "SET_TYPING", payload: true })

      // Simulate streaming response
      setTimeout(() => {
        const responses = [
          "I understand your question. Let me help you with that.",
          "That's an interesting point. Here's what I think about it...",
          "Based on the information you've provided, I can suggest the following approach:",
          "Let me break this down for you step by step.",
          "I'd be happy to help you with this. Here's my analysis:",
        ]

        const response = responses[Math.floor(Math.random() * responses.length)]

        dispatch({
          type: "ADD_MESSAGE",
          payload: {
            sessionId: state.currentSessionId!,
            role: "assistant",
            content: response,
            metadata: {
              model: "gpt-4",
              tokens: Math.floor(Math.random() * 100) + 50,
              processingTime: Math.floor(Math.random() * 2000) + 500,
            },
          },
        })

        dispatch({ type: "SET_STREAMING", payload: false })
        dispatch({ type: "SET_TYPING", payload: false })
      }, 1500)
    },
    [state.currentSessionId]
  )

  const regenerateMessage = useCallback(
    async (messageId: string) => {
      if (!currentSession) return

      const messageIndex = currentSession.messages.findIndex((m) => m.id === messageId)
      if (messageIndex === -1) return

      dispatch({ type: "SET_STREAMING", payload: true })

      // Simulate regeneration
      setTimeout(() => {
        const newResponses = [
          "Let me provide a different perspective on this...",
          "Here's another way to approach this problem:",
          "I can offer an alternative solution:",
          "Looking at this from another angle:",
        ]

        const newResponse = newResponses[Math.floor(Math.random() * newResponses.length)]

        dispatch({
          type: "UPDATE_MESSAGE",
          payload: {
            sessionId: currentSession.id,
            messageId,
            updates: { content: newResponse },
          },
        })

        dispatch({ type: "SET_STREAMING", payload: false })
      }, 1000)
    },
    [currentSession]
  )

  const stopGeneration = useCallback(() => {
    dispatch({ type: "SET_STREAMING", payload: false })
    dispatch({ type: "SET_TYPING", payload: false })
  }, [])

  const exportChat = useCallback(
    (sessionId: string, format: "json" | "markdown" | "txt") => {
      const session = state.sessions.find((s) => s.id === sessionId)
      if (!session) return

      let content = ""

      switch (format) {
        case "json":
          content = JSON.stringify(session, null, 2)
          break
        case "markdown":
          content = `# ${session.title}\n\n`
          session.messages.forEach((msg) => {
            content += `## ${msg.role.toUpperCase()}\n${msg.content}\n\n`
          })
          break
        case "txt":
          content = `${session.title}\n${"=".repeat(session.title.length)}\n\n`
          session.messages.forEach((msg) => {
            content += `${msg.role.toUpperCase()}: ${msg.content}\n\n`
          })
          break
      }

      const blob = new Blob([content], { type: "text/plain" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${session.title}.${format}`
      a.click()
      URL.revokeObjectURL(url)
    },
    [state.sessions]
  )

  const importChat = useCallback((data: any) => {
    try {
      if (Array.isArray(data)) {
        data.forEach((session) => {
          dispatch({ type: "CREATE_SESSION", payload: { title: session.title } })
        })
      } else if (data.messages) {
        dispatch({ type: "CREATE_SESSION", payload: { title: data.title } })
      }
    } catch (error) {
      logger.error("Failed to import chat", error as Error, { component: "chat-provider" })
    }
  }, [])

  return (
    <ChatContext.Provider
      value={{
        state,
        dispatch,
        currentSession,
        sendMessage,
        regenerateMessage,
        stopGeneration,
        exportChat,
        importChat,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider")
  }
  return context
}
