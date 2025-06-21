"use client"

import * as React from "react"
import {
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  MoreHorizontal,
  Edit,
  Trash2,
  Share,
  Bookmark,
  Check,
  CheckCheck,
  Clock,
  AlertCircle,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { useChat } from "@/hooks/chat-provider"
import { useSettings } from "@/hooks/settings-provider"
import { formatDistanceToNow, format } from "date-fns"
import { cn } from "@/lib/utils"
import { useToast } from "@/hooks/use-toast"
import type { MessageBubbleProps } from "@/lib/types/ui"
import { renderMarkdown } from "@/lib/utils/markdown"

// Remove duplicate interface as it now comes from centralized types

export function MessageBubble({ message, isLast }: MessageBubbleProps) {
  const { dispatch, regenerateMessage, currentSession } = useChat()
  const { settings } = useSettings()
  const { toast } = useToast()
  const [isHovered, setIsHovered] = React.useState(false)

  const isUser = message.role === "user"
  const isSystem = message.role === "system"

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      toast({
        title: "Copied to clipboard",
        description: "Message content has been copied to your clipboard.",
      })
    } catch (error) {
      toast({
        title: "Failed to copy",
        description: "Could not copy message to clipboard.",
        variant: "destructive",
      })
    }
  }

  const handleReaction = (type: "thumbs_up" | "thumbs_down") => {
    if (!currentSession) return

    const existingReaction = message.reactions?.find((r) => r.type === type)
    const updatedReactions = existingReaction
      ? message.reactions?.filter((r) => r.type !== type) || []
      : [
          ...(message.reactions || []),
          {
            id: `${Date.now()}`,
            user_id: "current_user",
            type,
            emoji: type === "thumbs_up" ? "👍" : "👎",
            created_at: new Date().toISOString(),
          },
        ]

    dispatch({
      type: "UPDATE_MESSAGE",
      payload: {
        sessionId: currentSession.id,
        messageId: message.id,
        updates: { reactions: updatedReactions },
      },
    })
  }

  const handleRegenerate = () => {
    regenerateMessage(message.id)
  }

  const handleDelete = () => {
    if (!currentSession) return
    dispatch({
      type: "DELETE_MESSAGE",
      payload: {
        sessionId: currentSession.id,
        messageId: message.id,
      },
    })
  }

  const getStatusIcon = () => {
    const statusString = message.status?.status || "sent"
    switch (statusString) {
      case "sending":
        return <Clock className="h-3 w-3 text-muted-foreground animate-spin" />
      case "sent":
        return <Check className="h-3 w-3 text-muted-foreground" />
      case "delivered":
        return <CheckCheck className="h-3 w-3 text-muted-foreground" />
      case "read":
        return <CheckCheck className="h-3 w-3 text-blue-500" />
      case "error":
        return <AlertCircle className="h-3 w-3 text-destructive" />
      default:
        return null
    }
  }

  const formatTimestamp = (date: Date) => {
    if (settings.showTimestamps) {
      return format(date, "HH:mm")
    }
    return formatDistanceToNow(date, { addSuffix: true })
  }

  return (
    <div
      className={cn(
        "group flex gap-3 message-enter",
        isUser ? "justify-end" : "justify-start",
        isSystem && "justify-center"
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {!isSystem && !isUser && settings.showAvatars && (
        <Avatar className="h-8 w-8 flex-shrink-0">
          <AvatarImage src="/placeholder.svg?height=32&width=32" />
          <AvatarFallback>AI</AvatarFallback>
        </Avatar>
      )}

      <div className={cn("flex flex-col gap-1 max-w-[80%]", isUser && "order-first")}>
        <div
          className={cn(
            "rounded-lg px-4 py-2 text-sm",
            isUser && "bg-primary text-primary-foreground ml-auto",
            !isUser && !isSystem && "bg-muted",
            isSystem && "bg-[hsl(var(--chat-system-bg))] text-[hsl(var(--chat-system-fg))] border text-center"
          )}
        >
          {!isUser && !isSystem ? (
            <div 
              className="prose-chat"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
            />
          ) : (
            <div className="whitespace-pre-wrap break-words">{message.content}</div>
          )}

          {message.attachments && message.attachments.length > 0 && (
            <div className="mt-2 space-y-2">
              {message.attachments.map((attachment) => (
                <div key={attachment.id} className="flex items-center gap-2 p-2 bg-muted/50 rounded">
                  <div className="text-xs">
                    📎 {attachment.name} ({(attachment.size / 1024).toFixed(1)}KB)
                  </div>
                </div>
              ))}
            </div>
          )}

          {message.reactions && message.reactions.length > 0 && (
            <div className="flex gap-1 mt-2">
              {message.reactions.map((reaction, index) => (
                <Badge key={index} variant="secondary" className="text-xs">
                  {reaction.type === "thumbs_up" ? "👍" : "👎"}
                </Badge>
              ))}
            </div>
          )}
        </div>

        <div className={cn("flex items-center gap-2 text-xs text-muted-foreground", isUser && "order-first")}>
          {settings.showTimestamps && message.timestamp && <span>{formatTimestamp(message.timestamp)}</span>}

          {isUser && getStatusIcon()}

          {message.metadata && (
            <div className="flex items-center gap-1">
              {message.metadata.model && (
                <Badge variant="outline" className="text-xs">
                  {message.metadata.model}
                </Badge>
              )}
              {message.metadata.tokens && <span className="text-xs">{message.metadata.tokens} tokens</span>}
            </div>
          )}
        </div>
      </div>

      {!isSystem && (isHovered || isLast) && (
        <div
          className={cn(
            "flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity",
            isUser && "order-first"
          )}
        >
          <Button variant="ghost" size="sm" onClick={handleCopy} className="h-8 w-8 p-0">
            <Copy className="h-3 w-3" />
          </Button>

          {!isUser && (
            <>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleReaction("thumbs_up")}
                className={cn(
                  "h-8 w-8 p-0",
                  message.reactions?.some((r) => r.type === "thumbs_up") && "text-green-500"
                )}
              >
                <ThumbsUp className="h-3 w-3" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleReaction("thumbs_down")}
                className={cn(
                  "h-8 w-8 p-0",
                  message.reactions?.some((r) => r.type === "thumbs_down") && "text-red-500"
                )}
              >
                <ThumbsDown className="h-3 w-3" />
              </Button>

              <Button variant="ghost" size="sm" onClick={handleRegenerate} className="h-8 w-8 p-0">
                <RotateCcw className="h-3 w-3" />
              </Button>
            </>
          )}

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <MoreHorizontal className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align={isUser ? "end" : "start"}>
              <DropdownMenuItem onClick={handleCopy}>
                <Copy className="h-4 w-4 mr-2" />
                Copy
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Share className="h-4 w-4 mr-2" />
                Share
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Bookmark className="h-4 w-4 mr-2" />
                Bookmark
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Edit className="h-4 w-4 mr-2" />
                Edit
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleDelete} className="text-destructive">
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}

      {isUser && settings.showAvatars && (
        <Avatar className="h-8 w-8 flex-shrink-0">
          <AvatarImage src="/placeholder.svg?height=32&width=32" />
          <AvatarFallback>U</AvatarFallback>
        </Avatar>
      )}
    </div>
  )
}
