"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { Loader2 } from "lucide-react"
import { useAuthStore } from "@/lib/stores/auth-store"
import { useChatStore } from "@/lib/stores/chat-store"
import { ChatInterface } from "@/components/chat/chat-interface"

// export const metadata: Metadata = {
//   title: "Archyx AI",
//   description: "Your AI assistant for Google Drive",
//   keywords: ["chat", "AI"],
//   openGraph: {
//     title: "Archyx AI",
//     description: "Your AI assistant for Google Drive",
//     url: "/",
//     type: "website",
//   },
// }

export default function HomePage() {
  const router = useRouter()
  const { user, loading, initialized, initialize } = useAuthStore()
  const { clearCurrentChat } = useChatStore()

  useEffect(() => {
    if (!initialized) {
      initialize()
    }
  }, [initialized, initialize])

  useEffect(() => {
    if (initialized && !loading) {
      if (!user) {
        router.push("/auth")
      } else {
        clearCurrentChat()
      }
    }
  }, [user, loading, initialized, router, clearCurrentChat])

  if (!initialized || loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="flex items-center gap-2">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span>Loading...</span>
        </div>
      </div>
    )
  }

  return <ChatInterface />
}
