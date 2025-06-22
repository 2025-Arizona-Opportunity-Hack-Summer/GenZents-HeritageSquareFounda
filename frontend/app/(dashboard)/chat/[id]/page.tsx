"use client"

import { useEffect } from "react"
import { useParams } from "next/navigation"
// import { Metadata } from "next"
import { useChatStore } from "@/lib/stores/chat-store"
import { ChatInterface } from "@/components/chat/chat-interface"

// export const metadata: Metadata = {
//   title: "Chat",
//   description: "Your AI assistant for Google Drive",
//   keywords: ["chat", "AI"],
//   openGraph: {
//     title: "Chat",
//     description: "Your AI assistant for Google Drive",
//     url: "/chat",
//     type: "website",
//   },
// }

export default function ChatPage() {
  const params = useParams()
  const chatId = params.id as string
  const { selectChat, getCurrentChat, loadChats } = useChatStore()

  useEffect(() => {
    loadChats()
  }, [loadChats])

  useEffect(() => {
    if (chatId) {
      selectChat(chatId)
    }
  }, [chatId, selectChat])

  const currentChat = getCurrentChat()

  if (!currentChat) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Chat not found</h2>
          <p className="text-muted-foreground">The requested chat could not be loaded.</p>
        </div>
      </div>
    )
  }

  return <ChatInterface />
}
