"use client"

import { ChatMessages } from "./chat-messages"
import { ChatInput } from "./chat-input"
import { ChatHeader } from "./chat-header"
import { useChatStore } from "@/lib/stores/chat-store"

export function ChatInterface() {
  const { getCurrentChat } = useChatStore()
  const currentChat = getCurrentChat()

  if (!currentChat) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <h1 className="text-3xl font-bold mb-4">Welcome to Archyx AI</h1>
            <p className="text-muted-foreground mb-6">
              Start a conversation by typing your message below. A new chat will be created automatically.
            </p>
          </div>
        </div>
        <ChatInput />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <ChatHeader />
      <div className="flex-1 overflow-hidden">
        <ChatMessages />
      </div>
      <ChatInput />
    </div>
  )
}
