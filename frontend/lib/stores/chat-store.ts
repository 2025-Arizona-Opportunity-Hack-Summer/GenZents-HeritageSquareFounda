import { create } from "zustand"
import { persist } from "zustand/middleware"
import { createClient } from "@/lib/supabase/client"
import { nanoid } from "nanoid"
import { AIService } from "@/lib/services/ai-service"
import { logger } from "@/lib/utils/logger"
import type { Chat, Message, Attachment, ContextFile } from "@/lib/types/chat"

// Helper function to generate chat title from message content
const generateChatTitle = (content: string): string => {
  // Remove any leading/trailing whitespace and normalize line breaks
  let cleaned = content.trim().replace(/\s+/g, ' ')
  
  // Remove basic markdown formatting
  cleaned = cleaned
    .replace(/\*\*(.*?)\*\*/g, '$1') // Bold
    .replace(/\*(.*?)\*/g, '$1')     // Italic
    .replace(/`(.*?)`/g, '$1')       // Inline code
    .replace(/#{1,6}\s/g, '')        // Headers
  
  // Split into words and take first 4-6 words
  const words = cleaned.split(/\s+/).filter(word => word.length > 0)
  const titleWords = words.slice(0, Math.min(6, words.length))
  
  // Join words and limit to 50 characters
  let title = titleWords.join(' ')
  if (title.length > 50) {
    title = title.substring(0, 47) + '...'
  }
  
  // Ensure we have a fallback title
  return title || "New Chat"
}

// Remove duplicate interfaces as they now come from centralized types

interface ChatState {
  chats: Chat[]
  currentChatId: string | null
  loading: boolean
  isStreaming: boolean
  searchQuery: string
  contextFiles: ContextFile[]
  attachments: Attachment[]

  // Actions
  loadChats: () => Promise<void>
  createChat: (title?: string, template?: Partial<Chat>) => Promise<string>
  selectChat: (chatId: string) => void
  clearCurrentChat: () => void
  deleteChat: (chatId: string) => Promise<void>
  updateChat: (chatId: string, updates: Partial<Chat>) => Promise<void>
  duplicateChat: (chatId: string) => Promise<string>
  archiveChat: (chatId: string) => Promise<void>

  // Messages
  sendMessage: (content: string, chatId?: string, parentId?: string) => Promise<string>
  sendMessageStream: (content: string, chatId?: string) => Promise<string>
  editMessage: (messageId: string, content: string) => Promise<void>
  deleteMessage: (messageId: string) => Promise<void>
  reactToMessage: (messageId: string, reaction: "thumbs_up" | "thumbs_down") => Promise<void>
  regenerateMessage: (messageId: string) => Promise<void>

  // Search
  searchMessages: (query: string) => Promise<Message[]>
  setSearchQuery: (query: string) => void

  // Context
  addContextFile: (file: ContextFile) => void
  removeContextFile: (fileId: string) => void
  clearContext: () => void

  // Attachments
  addAttachment: (attachment: Attachment) => void
  removeAttachment: (attachmentId: string) => void
  clearAttachments: () => void

  // Export/Import
  exportChat: (chatId: string, format: "json" | "markdown" | "txt") => Promise<string>
  importChat: (data: string, format: "json") => Promise<void>

  // Getters
  getCurrentChat: () => Chat | null
  getFilteredChats: (filter: "all" | "bookmarked" | "archived" | "shared") => Chat[]
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      chats: [],
      currentChatId: null,
      loading: false,
      isStreaming: false,
      searchQuery: "",
      contextFiles: [],
      attachments: [],

      loadChats: async () => {
        try {
          set({ loading: true })
          const supabase = createClient()

          const { data: chats, error } = await supabase
            .from("chats")
            .select(
              `
              *,
              messages (*)
            `
            )
            .order("updated_at", { ascending: false })

          if (error) throw error

          set({
            chats:
              chats?.map((chat) => ({
                ...chat,
                messages: chat.messages || [],
              })) || [],
            loading: false,
          })
        } catch (error) {
          logger.error("Error loading chats", error as Error, { component: "chat-store" })
          set({ loading: false })
        }
      },

      createChat: async (title = "New Chat", template = {}) => {
        try {
          const supabase = createClient()

          // Create chat using AIService which handles backend API call
          const chatId = await AIService.createChat(title, template.system_prompt)

          const newChat: Partial<Chat> = {
            id: chatId,
            title,
            user_id: (await supabase.auth.getUser()).data.user?.id!,
            bookmarked: false,
            archived: false,
            shared: false,
            version: 1,
            ...template,
          }

          const chat: Chat = {
            ...(newChat as Chat),
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            messages: [],
          }

          set((state) => ({
            chats: [chat, ...state.chats],
            currentChatId: chatId,
          }))

          return chatId
        } catch (error) {
          logger.error("Error creating chat", error as Error, { component: "chat-store" })
          throw error
        }
      },

      selectChat: (chatId: string) => {
        set({ currentChatId: chatId })
      },

      clearCurrentChat: () => {
        set({ currentChatId: null })
      },

      deleteChat: async (chatId: string) => {
        try {
          const supabase = createClient()

          // Delete messages first
          await supabase.from("messages").delete().eq("chat_id", chatId)

          // Delete chat
          const { error } = await supabase.from("chats").delete().eq("id", chatId)
          if (error) throw error

          set((state) => ({
            chats: state.chats.filter((chat) => chat.id !== chatId),
            currentChatId: state.currentChatId === chatId ? null : state.currentChatId,
          }))
        } catch (error) {
          logger.error("Error deleting chat", error as Error, { component: "chat-store", chatId })
          throw error
        }
      },

      updateChat: async (chatId: string, updates: Partial<Chat>) => {
        try {
          const supabase = createClient()

          const { error } = await supabase
            .from("chats")
            .update({ ...updates, updated_at: new Date().toISOString() })
            .eq("id", chatId)

          if (error) throw error

          set((state) => ({
            chats: state.chats.map((chat) => (chat.id === chatId ? { ...chat, ...updates } : chat)),
          }))
        } catch (error) {
          logger.error("Error updating chat", error as Error, { component: "chat-store", chatId })
          throw error
        }
      },

      duplicateChat: async (chatId: string) => {
        try {
          const chat = get().chats.find((c) => c.id === chatId)
          if (!chat) throw new Error("Chat not found")

          const newChatId = await get().createChat(`${chat.title} (Copy)`, {
            model: chat.model,
            system_prompt: chat.system_prompt,
            tags: chat.tags,
          })

          // Copy messages
          const supabase = createClient()
          const messagesToCopy = chat.messages.map((msg) => ({
            ...msg,
            id: nanoid(),
            chat_id: newChatId,
          }))

          if (messagesToCopy.length > 0) {
            await supabase.from("messages").insert(messagesToCopy)
          }

          return newChatId
        } catch (error) {
          logger.error("Error duplicating chat", error as Error, { component: "chat-store", chatId })
          throw error
        }
      },

      archiveChat: async (chatId: string) => {
        await get().updateChat(chatId, { archived: true })
      },

      sendMessage: async (content: string, chatId?: string, parentId?: string) => {
        try {
          let targetChatId = chatId || get().currentChatId

          if (!targetChatId) {
            targetChatId = await get().createChat()
          }

          set({ isStreaming: true })

          // Create user message for local state
          const userMessage: Message = {
            id: nanoid(),
            chat_id: targetChatId,
            role: "user",
            content,
            created_at: new Date().toISOString(),
            edited: false,
            deleted: false,
            metadata: {
              model: undefined,
              tokens: undefined,
              cost: undefined,
              processingTime: undefined,
              confidence: undefined,
              sources: [],
              citations: [],
              language: undefined,
              sentiment: undefined,
              topics: [],
              entities: [],
            },
            attachments: get().attachments,
            reactions: [],
            mentions: [],
            parentId,
            status: {
              sent: false,
              delivered: false,
              read: false,
              retries: 0,
              status: "sending",
            },
            encryption: {
              encrypted: false,
            },
          }

          // Update local state with user message
          set((state) => ({
            chats: state.chats.map((chat) =>
              chat.id === targetChatId
                ? {
                    ...chat,
                    messages: [
                      ...chat.messages,
                      {
                        ...userMessage,
                        status: {
                          ...userMessage.status,
                          sent: true,
                          status: "sent",
                        },
                      },
                    ],
                  }
                : chat
            ),
            attachments: [],
          }))

          // Check if this is the first message in the chat and update title
          const currentState = get()
          const targetChat = currentState.chats.find(chat => chat.id === targetChatId)
          if (targetChat && targetChat.messages.length === 1 && targetChat.title === "New Chat") {
            const newTitle = generateChatTitle(content)
            logger.info("Updating chat title based on first message", { 
              component: "chat-store", 
              chatId: targetChatId, 
              newTitle 
            })
            
            // Update title in local state
            set((state) => ({
              chats: state.chats.map((chat) =>
                chat.id === targetChatId
                  ? { ...chat, title: newTitle }
                  : chat
              ),
            }))
            
            // Update title in database (fire and forget)
            get().updateChat(targetChatId, { title: newTitle }).catch(error => 
              logger.error("Failed to update chat title in database", error, { 
                component: "chat-store", 
                chatId: targetChatId 
              })
            )
          }

          // Send message to backend using AIService
          const aiMessage = await AIService.sendMessage(targetChatId, content)

          // Add AI response to local state
          set((state) => ({
            chats: state.chats.map((chat) =>
              chat.id === targetChatId
                ? {
                    ...chat,
                    messages: [
                      ...chat.messages,
                      {
                        id: aiMessage.id,
                        chat_id: targetChatId,
                        role: "assistant",
                        content: aiMessage.content,
                        created_at: aiMessage.created_at,
                        edited: false,
                        deleted: false,
                        metadata: {
                          model: "gemini-1.5-flash",
                          tokens: undefined,
                          cost: undefined,
                          processingTime: undefined,
                          confidence: undefined,
                          sources: [],
                          citations: [],
                          language: undefined,
                          sentiment: undefined,
                          topics: [],
                          entities: [],
                        },
                        attachments: [],
                        reactions: [],
                        mentions: [],
                        status: {
                          sent: true,
                          delivered: true,
                          read: false,
                          retries: 0,
                          status: "sent",
                        },
                        encryption: {
                          encrypted: false,
                        },
                      },
                    ],
                  }
                : chat
            ),
            isStreaming: false,
          }))

          return targetChatId
        } catch (error) {
          logger.error("Error sending message", error as Error, { component: "chat-store", chatId })
          set({ isStreaming: false })
          throw error
        }
      },

      sendMessageStream: async (content: string, chatId?: string) => {
        try {
          let targetChatId = chatId || get().currentChatId

          if (!targetChatId) {
            logger.info("Creating new chat for message stream", { component: "chat-store" })
            targetChatId = await get().createChat()
            
            // Wait a moment for the chat to be fully created
            await new Promise(resolve => setTimeout(resolve, 100))
          }

          // Validate that we have a proper chat ID
          if (!targetChatId) {
            throw new Error("Failed to create or get chat ID")
          }

          logger.info("Starting message stream", { component: "chat-store", chatId: targetChatId, content: content.substring(0, 50) + "..." })

          set({ isStreaming: true })

          // Create user message for local state
          const userMessage: Message = {
            id: nanoid(),
            chat_id: targetChatId,
            role: "user",
            content,
            created_at: new Date().toISOString(),
            edited: false,
            deleted: false,
            metadata: {
              model: undefined,
              tokens: undefined,
              cost: undefined,
              processingTime: undefined,
              confidence: undefined,
              sources: [],
              citations: [],
              language: undefined,
              sentiment: undefined,
              topics: [],
              entities: [],
            },
            attachments: get().attachments,
            reactions: [],
            mentions: [],
            status: {
              sent: true,
              delivered: true,
              read: false,
              retries: 0,
              status: "sent",
            },
            encryption: {
              encrypted: false,
            },
          }

          // Add user message to local state
          set((state) => ({
            chats: state.chats.map((chat) =>
              chat.id === targetChatId
                ? {
                    ...chat,
                    messages: [...chat.messages, userMessage],
                  }
                : chat
            ),
            attachments: [],
          }))

          // Check if this is the first message in the chat and update title
          const currentState = get()
          const targetChat = currentState.chats.find(chat => chat.id === targetChatId)
          if (targetChat && targetChat.messages.length === 1 && targetChat.title === "New Chat") {
            const newTitle = generateChatTitle(content)
            logger.info("Updating chat title based on first message", { 
              component: "chat-store", 
              chatId: targetChatId, 
              newTitle 
            })
            
            // Update title in local state
            set((state) => ({
              chats: state.chats.map((chat) =>
                chat.id === targetChatId
                  ? { ...chat, title: newTitle }
                  : chat
              ),
            }))
            
            // Update title in database (fire and forget)
            get().updateChat(targetChatId, { title: newTitle }).catch(error => 
              logger.error("Failed to update chat title in database", error, { 
                component: "chat-store", 
                chatId: targetChatId 
              })
            )
          }

          // Create streaming AI message placeholder
          const aiMessageId = nanoid()
          const aiMessage: Message = {
            id: aiMessageId,
            chat_id: targetChatId,
            role: "assistant",
            content: "",
            created_at: new Date().toISOString(),
            edited: false,
            deleted: false,
            metadata: {
              model: "gemini-1.5-flash",
              tokens: undefined,
              cost: undefined,
              processingTime: undefined,
              confidence: undefined,
              sources: [],
              citations: [],
              language: undefined,
              sentiment: undefined,
              topics: [],
              entities: [],
            },
            attachments: [],
            reactions: [],
            mentions: [],
            status: {
              sent: false,
              delivered: false,
              read: false,
              retries: 0,
              status: "sending",
            },
            encryption: {
              encrypted: false,
            },
          }

          // Add AI message placeholder
          set((state) => ({
            chats: state.chats.map((chat) =>
              chat.id === targetChatId
                ? {
                    ...chat,
                    messages: [...chat.messages, aiMessage],
                  }
                : chat
            ),
          }))

          // Start streaming response
          try {
            logger.info("Calling AIService.sendMessageStream", { component: "chat-store", chatId: targetChatId })
            const stream = await AIService.sendMessageStream(targetChatId, content)
            logger.info("Got stream response, starting to parse", { component: "chat-store", chatId: targetChatId })

            for await (const response of AIService.parseStreamingResponse(stream)) {
              logger.debug("Received streaming response", { component: "chat-store", responseType: response.type })
              
              if (response.type === "chunk" && response.content) {
                // Update the AI message content with streaming chunks
                set((state) => ({
                  chats: state.chats.map((chat) =>
                    chat.id === targetChatId
                      ? {
                          ...chat,
                          messages: chat.messages.map((msg) =>
                            msg.id === aiMessageId
                              ? {
                                  ...msg,
                                  content: msg.content + response.content,
                                }
                              : msg
                          ),
                        }
                      : chat
                  ),
                }))
              } else if (response.type === "complete" && response.message) {
                logger.info("Streaming complete", { component: "chat-store", messageId: response.message.id })
                // Update with final message data from server
                set((state) => ({
                  chats: state.chats.map((chat) =>
                    chat.id === targetChatId
                      ? {
                          ...chat,
                          messages: chat.messages.map((msg) =>
                            msg.id === aiMessageId
                              ? {
                                  ...msg,
                                  id: response.message.id,
                                  content: response.message.content,
                                  created_at: response.message.created_at,
                                  status: {
                                    ...msg.status,
                                    sent: true,
                                    delivered: true,
                                    status: "sent",
                                  },
                                }
                              : msg
                          ),
                        }
                      : chat
                  ),
                  isStreaming: false,
                }))
                break
              } else if (response.type === "error") {
                logger.error("Streaming error received", undefined, { component: "chat-store", error: response.error })
                throw new Error(response.error || "Failed to generate response")
              }
            }
          } catch (streamError) {
            logger.warn("Streaming failed, falling back to regular message", { component: "chat-store", streamError })

            // Remove the placeholder AI message
            set((state) => ({
              chats: state.chats.map((chat) =>
                chat.id === targetChatId
                  ? {
                      ...chat,
                      messages: chat.messages.filter((msg) => msg.id !== aiMessageId),
                    }
                  : chat
              ),
            }))

            // Fall back to regular message sending
            try {
              logger.info("Attempting fallback to regular message sending", { component: "chat-store", chatId: targetChatId })
              const aiMessage = await AIService.sendMessage(targetChatId, content)
              logger.info("Fallback message sent successfully", { component: "chat-store", messageId: aiMessage.id })

              // Add the regular AI response
              set((state) => ({
                chats: state.chats.map((chat) =>
                  chat.id === targetChatId
                    ? {
                        ...chat,
                        messages: [
                          ...chat.messages,
                          {
                            id: aiMessage.id,
                            chat_id: targetChatId,
                            role: "assistant",
                            content: aiMessage.content,
                            created_at: aiMessage.created_at,
                            edited: false,
                            deleted: false,
                            metadata: {
                              model: "gemini-1.5-flash",
                              tokens: undefined,
                              cost: undefined,
                              processingTime: undefined,
                              confidence: undefined,
                              sources: [],
                              citations: [],
                              language: undefined,
                              sentiment: undefined,
                              topics: [],
                              entities: [],
                            },
                            attachments: [],
                            reactions: [],
                            mentions: [],
                            status: {
                              sent: true,
                              delivered: true,
                              read: false,
                              retries: 0,
                              status: "sent",
                            },
                            encryption: {
                              encrypted: false,
                            },
                          },
                        ],
                      }
                    : chat
                ),
                isStreaming: false,
              }))
            } catch (fallbackError) {
              logger.error("Both streaming and regular message failed", fallbackError as Error, { component: "chat-store", chatId: targetChatId })
              set({ isStreaming: false })
              throw fallbackError
            }
          }

          return targetChatId
        } catch (error) {
          logger.error("Error sending streaming message", error as Error, { component: "chat-store", chatId })
          set({ isStreaming: false })
          throw error
        }
      },

      editMessage: async (messageId: string, content: string) => {
        try {
          const supabase = createClient()

          const { error } = await supabase
            .from("messages")
            .update({
              content,
              updated_at: new Date().toISOString(),
              metadata: { edited: true },
            })
            .eq("id", messageId)

          if (error) throw error

          set((state) => ({
            chats: state.chats.map((chat) => ({
              ...chat,
              messages: chat.messages.map((msg) =>
                msg.id === messageId ? { ...msg, content, edited: true, updated_at: new Date().toISOString() } : msg
              ),
            })),
          }))
        } catch (error) {
          logger.error("Error editing message", error as Error, { component: "chat-store", messageId })
          throw error
        }
      },

      deleteMessage: async (messageId: string) => {
        try {
          const supabase = createClient()

          const { error } = await supabase.from("messages").delete().eq("id", messageId)
          if (error) throw error

          set((state) => ({
            chats: state.chats.map((chat) => ({
              ...chat,
              messages: chat.messages.filter((msg) => msg.id !== messageId),
            })),
          }))
        } catch (error) {
          logger.error("Error deleting message", error as Error, { component: "chat-store", messageId })
          throw error
        }
      },

      reactToMessage: async (messageId: string, reaction: "thumbs_up" | "thumbs_down") => {
        try {
          const supabase = createClient()
          const userId = (await supabase.auth.getUser()).data.user?.id

          if (!userId) throw new Error("User not authenticated")

          set((state) => ({
            chats: state.chats.map((chat) => ({
              ...chat,
              messages: chat.messages.map((msg) => {
                if (msg.id === messageId) {
                  const reactions = msg.reactions || []
                  const existingReaction = reactions.find((r) => r.user_id === userId)

                  if (existingReaction) {
                    return {
                      ...msg,
                      reactions: reactions.filter((r) => r.user_id !== userId),
                    }
                  } else {
                    return {
                      ...msg,
                      reactions: [
                        ...reactions,
                        {
                          id: nanoid(),
                          user_id: userId,
                          type: reaction,
                          emoji: reaction === "thumbs_up" ? "👍" : "👎",
                          created_at: new Date().toISOString(),
                        },
                      ],
                    }
                  }
                }
                return msg
              }),
            })),
          }))
        } catch (error) {
          logger.error("Error reacting to message", error as Error, { component: "chat-store", messageId, reaction })
          throw error
        }
      },

      regenerateMessage: async (messageId: string) => {
        try {
          const chat = get().getCurrentChat()
          if (!chat) return

          const messageIndex = chat.messages.findIndex((m) => m.id === messageId)
          if (messageIndex === -1) return

          const previousMessage = chat.messages[messageIndex - 1]
          if (!previousMessage || previousMessage.role !== "user") return

          // Delete the current AI message
          await get().deleteMessage(messageId)

          // Regenerate response
          await get().sendMessage(previousMessage.content, chat.id)
        } catch (error) {
          logger.error("Error regenerating message", error as Error, { component: "chat-store", messageId })
          throw error
        }
      },

      searchMessages: async (query: string) => {
        try {
          const supabase = createClient()

          const { data: messages, error } = await supabase
            .from("messages")
            .select("*")
            .textSearch("content", query)
            .limit(50)

          if (error) throw error
          return messages || []
        } catch (error) {
          logger.error("Error searching messages", error as Error, { component: "chat-store", query })
          return []
        }
      },

      setSearchQuery: (query: string) => {
        set({ searchQuery: query })
      },

      addContextFile: (file: ContextFile) => {
        set((state) => ({
          contextFiles: [...state.contextFiles, file],
        }))
      },

      removeContextFile: (fileId: string) => {
        set((state) => ({
          contextFiles: state.contextFiles.filter((f) => f.id !== fileId),
        }))
      },

      clearContext: () => {
        set({ contextFiles: [] })
      },

      addAttachment: (attachment: Attachment) => {
        set((state) => ({
          attachments: [...state.attachments, attachment],
        }))
      },

      removeAttachment: (attachmentId: string) => {
        set((state) => ({
          attachments: state.attachments.filter((a) => a.id !== attachmentId),
        }))
      },

      clearAttachments: () => {
        set({ attachments: [] })
      },

      exportChat: async (chatId: string, format: "json" | "markdown" | "txt") => {
        try {
          const chat = get().chats.find((c) => c.id === chatId)
          if (!chat) throw new Error("Chat not found")

          switch (format) {
            case "json":
              return JSON.stringify(chat, null, 2)

            case "markdown":
              let markdown = `# ${chat.title}\n\n`
              markdown += `Created: ${new Date(chat.created_at).toLocaleString()}\n\n`

              chat.messages.forEach((msg) => {
                markdown += `## ${msg.role === "user" ? "User" : "Assistant"}\n\n`
                markdown += `${msg.content}\n\n`
                if (msg.created_at) {
                  markdown += `*${new Date(msg.created_at).toLocaleString()}*\n\n`
                }
              })

              return markdown

            case "txt":
              let text = `${chat.title}\n${"=".repeat(chat.title.length)}\n\n`

              chat.messages.forEach((msg) => {
                text += `[${msg.role.toUpperCase()}]: ${msg.content}\n\n`
              })

              return text

            default:
              throw new Error("Unsupported format")
          }
        } catch (error) {
          logger.error("Error exporting chat", error as Error, { component: "chat-store", chatId })
          throw error
        }
      },

      importChat: async (data: string, format: "json") => {
        try {
          if (format === "json") {
            const chatData = JSON.parse(data)
            const newChatId = await get().createChat(chatData.title, {
              model: chatData.model,
              system_prompt: chatData.system_prompt,
              tags: chatData.tags,
            })

            if (chatData.messages && chatData.messages.length > 0) {
              const supabase = createClient()
              const messagesToImport = chatData.messages.map((msg: Message) => ({
                ...msg,
                id: nanoid(),
                chat_id: newChatId,
              }))

              await supabase.from("messages").insert(messagesToImport)
            }
          }
        } catch (error) {
          logger.error("Error importing chat", error as Error, { component: "chat-store" })
          throw error
        }
      },

      getCurrentChat: () => {
        const { chats, currentChatId } = get()
        return chats.find((chat) => chat.id === currentChatId) || null
      },

      getFilteredChats: (filter: "all" | "bookmarked" | "archived" | "shared") => {
        const { chats } = get()

        switch (filter) {
          case "bookmarked":
            return chats.filter((chat) => chat.bookmarked)
          case "archived":
            return chats.filter((chat) => chat.archived)
          case "shared":
            return chats.filter((chat) => chat.shared)
          default:
            return chats.filter((chat) => !chat.archived)
        }
      },
    }),
    {
      name: "chat-storage",
      partialize: (state) => ({
        currentChatId: state.currentChatId,
        contextFiles: state.contextFiles,
      }),
    }
  )
)
