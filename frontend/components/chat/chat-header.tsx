"use client"

import { Button } from "@/components/ui/button"
import { MoreHorizontal, Edit, Bookmark, Share, Trash2 } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { useChatStore } from "@/lib/stores/chat-store"
import { useState } from "react"
import { Input } from "@/components/ui/input"
import { useRouter } from "next/navigation"

export function ChatHeader() {
  const { getCurrentChat, updateChat, deleteChat } = useChatStore()
  const [isEditing, setIsEditing] = useState(false)
  const [title, setTitle] = useState("")
  const router = useRouter()

  const currentChat = getCurrentChat()

  if (!currentChat) return null

  const handleEditTitle = () => {
    setTitle(currentChat.title)
    setIsEditing(true)
  }

  const handleSaveTitle = async () => {
    if (title.trim()) {
      await updateChat(currentChat.id, { title: title.trim() })
    }
    setIsEditing(false)
  }

  const handleBookmark = async () => {
    await updateChat(currentChat.id, { bookmarked: !currentChat.bookmarked })
  }

  const handleDelete = async () => {
    await deleteChat(currentChat.id)
    router.push("/")
  }

  return (
    <div className="flex items-center justify-between p-4 border-b">
      <div className="flex-1">
        {isEditing ? (
          <Input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onBlur={handleSaveTitle}
            onKeyDown={(e) => e.key === "Enter" && handleSaveTitle()}
            className="text-lg font-semibold"
            autoFocus
          />
        ) : (
          <h1 className="text-lg font-semibold truncate">{currentChat.title}</h1>
        )}
      </div>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm">
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={handleEditTitle}>
            <Edit className="h-4 w-4 mr-2" />
            Edit Title
          </DropdownMenuItem>
          <DropdownMenuItem onClick={handleBookmark}>
            <Bookmark className={`h-4 w-4 mr-2 ${currentChat.bookmarked ? "fill-current" : ""}`} />
            {currentChat.bookmarked ? "Remove Bookmark" : "Bookmark"}
          </DropdownMenuItem>
          <DropdownMenuItem>
            <Share className="h-4 w-4 mr-2" />
            Share
          </DropdownMenuItem>
          <DropdownMenuItem onClick={handleDelete} className="text-destructive">
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
