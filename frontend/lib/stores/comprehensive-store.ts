import { create } from "zustand"
import { persist, subscribeWithSelector } from "zustand/middleware"
import { supabase } from "@/lib/supabase/client"
import { logger } from "@/lib/utils/logger"
import type { Chat, Message, User, Workspace, AIModel, Analytics, Integration } from "@/lib/types"

interface ComprehensiveState {
  // Core Data
  user: User | null
  workspaces: Workspace[]
  currentWorkspace: Workspace | null
  chats: Chat[]
  currentChat: Chat | null
  messages: Message[]

  // AI & Models
  availableModels: AIModel[]
  currentModel: AIModel | null

  integrations: Integration[]

  // Analytics & Monitoring
  analytics: Analytics[]
  performance: PerformanceMetrics

  // UI State
  sidebarCollapsed: boolean
  theme: "light" | "dark" | "system"
  loading: Record<string, boolean>
  errors: Record<string, string>

  // Search & Filters
  searchQuery: string
  filters: SearchFilters

  // Real-time Features
  onlineUsers: string[]
  typingUsers: Record<string, string[]>
  chatChannels: Record<string, any>

  // Actions
  initialize: () => Promise<void>

  // User Management
  updateUser: (updates: Partial<User>) => Promise<void>
  updatePreferences: (preferences: Partial<User["preferences"]>) => Promise<void>

  // Workspace Management
  createWorkspace: (workspace: Partial<Workspace>) => Promise<string>
  updateWorkspace: (id: string, updates: Partial<Workspace>) => Promise<void>
  deleteWorkspace: (id: string) => Promise<void>
  switchWorkspace: (id: string) => void

  // Chat Management
  createChat: (chat: Partial<Chat>) => Promise<string>
  updateChat: (id: string, updates: Partial<Chat>) => Promise<void>
  deleteChat: (id: string) => Promise<void>
  duplicateChat: (id: string) => Promise<string>
  archiveChat: (id: string) => Promise<void>
  shareChat: (id: string, permissions: any) => Promise<string>

  // Message Management
  sendMessage: (content: string, options?: any) => Promise<void>
  editMessage: (id: string, content: string) => Promise<void>
  deleteMessage: (id: string) => Promise<void>
  reactToMessage: (id: string, reaction: string) => Promise<void>

  // Search & Discovery
  searchChats: (query: string) => Promise<Chat[]>
  searchMessages: (query: string) => Promise<Message[]>
  getRecommendations: () => Promise<Chat[]>

  // Integration Management
  addIntegration: (integration: Integration) => Promise<void>
  removeIntegration: (id: string) => Promise<void>

  // Analytics
  trackEvent: (event: string, properties?: any) => void
  getAnalytics: (timeRange: string) => Promise<Analytics[]>

  // AI Response Generation
  generateAIResponse: (chatId: string, userMessage: string) => Promise<void>

  // Real-time
  subscribeToChat: (chatId: string) => void
  unsubscribeFromChat: (chatId: string) => void
  updateTypingStatus: (chatId: string, typing: boolean) => void

  // Import/Export
  exportData: (format: "json" | "csv" | "pdf") => Promise<string>
  importData: (data: string, format: "json") => Promise<void>

  // Backup & Sync
  createBackup: () => Promise<string>
  restoreBackup: (backupId: string) => Promise<void>
  syncData: () => Promise<void>
}

interface PerformanceMetrics {
  responseTime: number
  throughput: number
  errorRate: number
  uptime: number
  memoryUsage: number
  cpuUsage: number
}

interface SearchFilters {
  dateRange: [string, string]
  users: string[]
  tags: string[]
  models: string[]
  status: string[]
}

export const useComprehensiveStore = create<ComprehensiveState>()(
  subscribeWithSelector(
    persist(
      (set, get) => ({
        // Initial State
        user: null,
        workspaces: [],
        currentWorkspace: null,
        chats: [],
        currentChat: null,
        messages: [],
        availableModels: [],
        currentModel: null,
        integrations: [],
        analytics: [],
        performance: {
          responseTime: 0,
          throughput: 0,
          errorRate: 0,
          uptime: 100,
          memoryUsage: 0,
          cpuUsage: 0,
        },
        sidebarCollapsed: false,
        theme: "system",
        loading: {},
        errors: {},
        searchQuery: "",
        filters: {
          dateRange: ["", ""],
          users: [],
          tags: [],
          models: [],
          status: [],
        },
        onlineUsers: [],
        typingUsers: {},
        chatChannels: {},

        // Initialize
        initialize: async () => {
          try {
            set({ loading: { ...get().loading, init: true } })

            // Load user data
            const {
              data: { user },
            } = await supabase.auth.getUser()
            if (user) {
              const { data: userData } = await supabase.from("users").select("*").eq("id", user.id).single()

              set({ user: userData })
            }

            // Load workspaces
            const { data: workspaces } = await supabase.from("workspaces").select("*").eq("owner_id", user?.id)

            set({ workspaces: workspaces || [] })

            // Load available models
            const { data: models } = await supabase.from("ai_models").select("*").eq("status", "active")

            set({ availableModels: models || [] })

            set({ loading: { ...get().loading, init: false } })
          } catch (error) {
            logger.error("Initialization error", error as Error, { component: "comprehensive-store" })
            set({
              loading: { ...get().loading, init: false },
              errors: { ...get().errors, init: "Failed to initialize" },
            })
          }
        },

        // User Management
        updateUser: async (updates) => {
          try {
            const { error } = await supabase.from("users").update(updates).eq("id", get().user?.id)

            if (error) throw error

            set({ user: { ...get().user!, ...updates } })
          } catch (error) {
            logger.error("Update user error", error as Error, { component: "comprehensive-store" })
          }
        },

        updatePreferences: async (preferences) => {
          const user = get().user
          if (!user) return

          const updatedPreferences = { ...user.preferences, ...preferences }
          await get().updateUser({ preferences: updatedPreferences })
        },

        // Workspace Management
        createWorkspace: async (workspace) => {
          try {
            const { data, error } = await supabase.from("workspaces").insert(workspace).select().single()

            if (error) throw error

            set({ workspaces: [...get().workspaces, data] })
            return data.id
          } catch (error) {
            logger.error("Create workspace error", error as Error, { component: "comprehensive-store" })
            throw error
          }
        },

        updateWorkspace: async (id, updates) => {
          try {
            const { error } = await supabase.from("workspaces").update(updates).eq("id", id)

            if (error) throw error

            set({
              workspaces: get().workspaces.map((w) => (w.id === id ? { ...w, ...updates } : w)),
            })
          } catch (error) {
            logger.error("Update workspace error", error as Error, { component: "comprehensive-store" })
          }
        },

        deleteWorkspace: async (id) => {
          try {
            const { error } = await supabase.from("workspaces").delete().eq("id", id)

            if (error) throw error

            set({
              workspaces: get().workspaces.filter((w) => w.id !== id),
              currentWorkspace: get().currentWorkspace?.id === id ? null : get().currentWorkspace,
            })
          } catch (error) {
            logger.error("Delete workspace error", error as Error, { component: "comprehensive-store" })
          }
        },

        switchWorkspace: (id) => {
          const workspace = get().workspaces.find((w) => w.id === id)
          if (workspace) {
            set({ currentWorkspace: workspace })
          }
        },

        // Chat Management
        createChat: async (chat) => {
          try {
            const { data, error } = await supabase
              .from("chats")
              .insert({
                ...chat,
                user_id: get().user?.id,
                workspace_id: get().currentWorkspace?.id,
              })
              .select()
              .single()

            if (error) throw error

            set({ chats: [...get().chats, data] })
            return data.id
          } catch (error) {
            logger.error("Create chat error", error as Error, { component: "comprehensive-store" })
            throw error
          }
        },

        updateChat: async (id, updates) => {
          try {
            const { error } = await supabase.from("chats").update(updates).eq("id", id)

            if (error) throw error

            set({
              chats: get().chats.map((c) => (c.id === id ? { ...c, ...updates } : c)),
            })
          } catch (error) {
            logger.error("Update chat error", error as Error, { component: "comprehensive-store" })
          }
        },

        deleteChat: async (id) => {
          try {
            const { error } = await supabase.from("chats").delete().eq("id", id)

            if (error) throw error

            set({
              chats: get().chats.filter((c) => c.id !== id),
              currentChat: get().currentChat?.id === id ? null : get().currentChat,
            })
          } catch (error) {
            logger.error("Delete chat error", error as Error, { component: "comprehensive-store" })
          }
        },

        duplicateChat: async (id) => {
          const chat = get().chats.find((c) => c.id === id)
          if (!chat) throw new Error("Chat not found")

          const { id: _, created_at, updated_at, ...chatData } = chat
          return await get().createChat({
            ...chatData,
            title: `${chat.title} (Copy)`,
          })
        },

        archiveChat: async (id) => {
          await get().updateChat(id, { status: "archived" })
        },

        shareChat: async (id, permissions) => {
          try {
            const { data, error } = await supabase
              .from("chat_shares")
              .insert({
                chat_id: id,
                permissions,
                created_by: get().user?.id,
              })
              .select()
              .single()

            if (error) throw error

            return data.share_token
          } catch (error) {
            logger.error("Share chat error", error as Error, { component: "comprehensive-store" })
            throw error
          }
        },

        // Message Management
        sendMessage: async (content, options = {}) => {
          try {
            const chatId = get().currentChat?.id
            if (!chatId) throw new Error("No active chat")

            const message = {
              chat_id: chatId,
              user_id: get().user?.id,
              role: "user",
              content,
              ...options,
            }

            const { data, error } = await supabase.from("messages").insert(message).select().single()

            if (error) throw error

            set({ messages: [...get().messages, data] })

            // Trigger AI response
            await get().generateAIResponse(chatId, content)
          } catch (error) {
            logger.error("Send message error", error as Error, { component: "comprehensive-store" })
          }
        },

        editMessage: async (id, content) => {
          try {
            const { error } = await supabase.from("messages").update({ content, edited: true }).eq("id", id)

            if (error) throw error

            set({
              messages: get().messages.map((m) => (m.id === id ? { ...m, content, edited: true } : m)),
            })
          } catch (error) {
            logger.error("Edit message error", error as Error, { component: "comprehensive-store" })
          }
        },

        deleteMessage: async (id) => {
          try {
            const { error } = await supabase.from("messages").update({ deleted: true }).eq("id", id)

            if (error) throw error

            set({
              messages: get().messages.map((m) => (m.id === id ? { ...m, deleted: true } : m)),
            })
          } catch (error) {
            logger.error("Delete message error", error as Error, { component: "comprehensive-store" })
          }
        },

        reactToMessage: async (id, reaction) => {
          try {
            const { error } = await supabase.from("message_reactions").insert({
              message_id: id,
              user_id: get().user?.id,
              type: reaction,
            })

            if (error) throw error

            // Update local state
            set({
              messages: get().messages.map((m) =>
                m.id === id
                  ? {
                      ...m,
                      reactions: [
                        ...(m.reactions || []),
                        {
                          id: Date.now().toString(),
                          user_id: get().user?.id!,
                          type: reaction,
                          emoji: reaction,
                          created_at: new Date().toISOString(),
                        },
                      ],
                    }
                  : m
              ),
            })
          } catch (error) {
            logger.error("React to message error", error as Error, { component: "comprehensive-store" })
          }
        },

        // Search & Discovery
        searchChats: async (query) => {
          try {
            const { data, error } = await supabase
              .from("chats")
              .select("*")
              .textSearch("title", query)
              .eq("user_id", get().user?.id)

            if (error) throw error
            return data || []
          } catch (error) {
            logger.error("Search chats error", error as Error, { component: "comprehensive-store" })
            return []
          }
        },

        searchMessages: async (query) => {
          try {
            const { data, error } = await supabase
              .from("messages")
              .select("*")
              .textSearch("content", query)
              .eq("user_id", get().user?.id)

            if (error) throw error
            return data || []
          } catch (error) {
            logger.error("Search messages error", error as Error, { component: "comprehensive-store" })
            return []
          }
        },

        getRecommendations: async () => {
          // Implement recommendation algorithm
          return []
        },

        // Integration Management
        addIntegration: async (integration) => {
          try {
            const { data, error } = await supabase
              .from("integrations")
              .insert({
                ...integration,
                user_id: get().user?.id,
              })
              .select()
              .single()

            if (error) throw error

            set({ integrations: [...get().integrations, data] })
          } catch (error) {
            logger.error("Add integration error", error as Error, { component: "comprehensive-store" })
          }
        },

        removeIntegration: async (id) => {
          try {
            const { error } = await supabase.from("integrations").delete().eq("id", id)

            if (error) throw error

            set({ integrations: get().integrations.filter((i) => i.id !== id) })
          } catch (error) {
            logger.error("Remove integration error", error as Error, { component: "comprehensive-store" })
          }
        },

        // Analytics
        trackEvent: (event, properties = {}) => {
          const user = get().user
          const workspace = get().currentWorkspace

          if (!user?.id) {
            logger.warn("Cannot track event without user ID", { event, component: "comprehensive-store" })
            return
          }

          const analyticsData: Analytics = {
            user_id: user.id,
            workspace_id: workspace?.id,
            event,
            properties,
            timestamp: new Date().toISOString(),
            session_id: sessionStorage.getItem("session_id") || "anonymous",
          }

          // Send to analytics service
          supabase.from("analytics").insert(analyticsData)

          set({ analytics: [...get().analytics, analyticsData] })
        },

        getAnalytics: async (timeRange) => {
          try {
            const { data, error } = await supabase
              .from("analytics")
              .select("*")
              .eq("user_id", get().user?.id)
              .gte("timestamp", timeRange)

            if (error) throw error
            return data || []
          } catch (error) {
            logger.error("Get analytics error", error as Error, { component: "comprehensive-store" })
            return []
          }
        },

        // Real-time
        subscribeToChat: (chatId) => {
          const channel = supabase
            .channel(`chat:${chatId}`)
            .on(
              "postgres_changes",
              {
                event: "INSERT",
                schema: "public",
                table: "messages",
                filter: `chat_id=eq.${chatId}`,
              },
              (payload) => {
                set({ messages: [...get().messages, payload.new as Message] })
              }
            )
            .subscribe()

          // Store channel reference for cleanup
          set({
            chatChannels: {
              ...get().chatChannels,
              [chatId]: channel,
            },
          })
        },

        unsubscribeFromChat: (chatId) => {
          const channel = get().chatChannels[chatId]
          if (channel) {
            supabase.removeChannel(channel)
            const { [chatId]: _, ...remainingChannels } = get().chatChannels
            set({ chatChannels: remainingChannels })
          }
        },

        updateTypingStatus: (chatId, typing) => {
          const channel = supabase.channel(`typing:${chatId}`)
          if (typing) {
            channel.send({
              type: "broadcast",
              event: "typing",
              payload: { user_id: get().user?.id, typing: true },
            })
          } else {
            channel.send({
              type: "broadcast",
              event: "typing",
              payload: { user_id: get().user?.id, typing: false },
            })
          }
        },

        // Import/Export
        exportData: async (format) => {
          const data = {
            user: get().user,
            chats: get().chats,
            messages: get().messages,
            workspaces: get().workspaces,
          }

          switch (format) {
            case "json":
              return JSON.stringify(data, null, 2)
            case "csv":
              // Convert to CSV format
              return "CSV export not implemented"
            case "pdf":
              // Generate PDF
              return "PDF export not implemented"
            default:
              throw new Error("Unsupported format")
          }
        },

        importData: async (data, format) => {
          if (format === "json") {
            const parsed = JSON.parse(data)
            // Merge imported data with existing data
            set({
              chats: [...get().chats, ...parsed.chats],
              messages: [...get().messages, ...parsed.messages],
            })
          }
        },

        // Backup & Sync
        createBackup: async () => {
          const backupData = await get().exportData("json")

          const { data, error } = await supabase
            .from("backups")
            .insert({
              user_id: get().user?.id,
              data: backupData,
              created_at: new Date().toISOString(),
            })
            .select()
            .single()

          if (error) throw error
          return data.id
        },

        restoreBackup: async (backupId) => {
          const { data, error } = await supabase.from("backups").select("data").eq("id", backupId).single()

          if (error) throw error

          await get().importData(data.data, "json")
        },

        syncData: async () => {
          // Implement data synchronization
          await get().initialize()
        },

        // AI Response Generation (placeholder)
        generateAIResponse: async (chatId: string, userMessage: string) => {
          // This would integrate with your AI service
          const aiResponse = {
            chat_id: chatId,
            user_id: "ai-assistant",
            role: "assistant" as const,
            content: `AI response to: ${userMessage}`,
            created_at: new Date().toISOString(),
            metadata: {
              model: get().currentModel?.id || "gpt-4",
              tokens: 150,
              processingTime: 1000,
            },
          }

          const { data, error } = await supabase.from("messages").insert(aiResponse).select().single()

          if (!error) {
            set({ messages: [...get().messages, data] })
          }
        },
      }),
      {
        name: "comprehensive-store",
        partialize: (state) => ({
          user: state.user,
          currentWorkspace: state.currentWorkspace,
          theme: state.theme,
          sidebarCollapsed: state.sidebarCollapsed,
        }),
      }
    )
  )
)
