"use client"

import * as React from "react"
import { MessageSquare, Search, Star, Trash2, MoreHorizontal, Download, BookmarkPlus, Clock } from "lucide-react"

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { useChat } from "@/hooks/chat-provider"
import { formatDistanceToNow } from "date-fns"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"

export function ChatHistory() {
  const { state, dispatch, exportChat } = useChat()
  const [searchQuery, setSearchQuery] = React.useState("")
  const [showArchived, setShowArchived] = React.useState(false)

  const filteredSessions = React.useMemo(() => {
    return state.sessions.filter((session) => {
      const matchesSearch =
        session.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        session.messages.some((msg) => msg.content.toLowerCase().includes(searchQuery.toLowerCase()))

      return matchesSearch
    })
  }, [state.sessions, searchQuery])

  const groupedSessions = React.useMemo(() => {
    const now = new Date()
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000)
    const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000)
    const lastMonth = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000)

    const groups = {
      today: [] as typeof filteredSessions,
      yesterday: [] as typeof filteredSessions,
      lastWeek: [] as typeof filteredSessions,
      lastMonth: [] as typeof filteredSessions,
      older: [] as typeof filteredSessions,
    }

    filteredSessions.forEach((session) => {
      const sessionDate = new Date(session.updatedAt)

      if (sessionDate >= today) {
        groups.today.push(session)
      } else if (sessionDate >= yesterday) {
        groups.yesterday.push(session)
      } else if (sessionDate >= lastWeek) {
        groups.lastWeek.push(session)
      } else if (sessionDate >= lastMonth) {
        groups.lastMonth.push(session)
      } else {
        groups.older.push(session)
      }
    })

    return groups
  }, [filteredSessions])

  const handleSelectSession = (sessionId: string) => {
    dispatch({ type: "SELECT_SESSION", payload: sessionId })
  }

  const handleDeleteSession = (sessionId: string) => {
    dispatch({ type: "DELETE_SESSION", payload: sessionId })
  }

  const handleBookmarkSession = (sessionId: string) => {
    dispatch({
      type: "UPDATE_SESSION",
      payload: {
        id: sessionId,
        updates: { bookmarked: true },
      },
    })
  }

  const handleExportSession = (sessionId: string, format: "json" | "markdown" | "txt") => {
    exportChat(sessionId, format)
  }

  const renderSessionGroup = (title: string, sessions: typeof filteredSessions) => {
    if (sessions.length === 0) return null

    return (
      <Collapsible key={title} defaultOpen={title === "today"}>
        <SidebarMenuSubItem>
          <CollapsibleTrigger asChild>
            <SidebarMenuSubButton className="w-full justify-between">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{title}</span>
              <Badge variant="secondary" className="text-xs">
                {sessions.length}
              </Badge>
            </SidebarMenuSubButton>
          </CollapsibleTrigger>
        </SidebarMenuSubItem>
        <CollapsibleContent>
          <SidebarMenuSub>
            {sessions.map((session) => (
              <SidebarMenuSubItem key={session.id}>
                <SidebarMenuSubButton asChild isActive={state.currentSessionId === session.id} className="group">
                  <button onClick={() => handleSelectSession(session.id)} className="w-full text-left">
                    <MessageSquare className="h-3 w-3" />
                    <div className="flex-1 min-w-0">
                      <div className="truncate text-sm">{session.title}</div>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        {formatDistanceToNow(session.updatedAt, { addSuffix: true })}
                        {session.bookmarked && <Star className="h-3 w-3 fill-current" />}
                      </div>
                    </div>
                  </button>
                </SidebarMenuSubButton>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <SidebarMenuAction className="opacity-0 group-hover:opacity-100">
                      <MoreHorizontal className="h-3 w-3" />
                    </SidebarMenuAction>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent side="right" align="start">
                    <DropdownMenuItem onClick={() => handleBookmarkSession(session.id)}>
                      <BookmarkPlus className="h-4 w-4" />
                      Bookmark
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => handleExportSession(session.id, "markdown")}>
                      <Download className="h-4 w-4" />
                      Export as Markdown
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleExportSession(session.id, "json")}>
                      <Download className="h-4 w-4" />
                      Export as JSON
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => handleDeleteSession(session.id)} className="text-destructive">
                      <Trash2 className="h-4 w-4" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </SidebarMenuSubItem>
            ))}
          </SidebarMenuSub>
        </CollapsibleContent>
      </Collapsible>
    )
  }

  return (
    <SidebarGroup>
      <SidebarGroupLabel>Chat History</SidebarGroupLabel>
      <SidebarGroupContent>
        <div className="px-2 pb-2">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search chats..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 h-8"
            />
          </div>
        </div>
        <SidebarMenu>
          <SidebarMenuSub>
            {renderSessionGroup("Today", groupedSessions.today)}
            {renderSessionGroup("Yesterday", groupedSessions.yesterday)}
            {renderSessionGroup("Last 7 days", groupedSessions.lastWeek)}
            {renderSessionGroup("Last 30 days", groupedSessions.lastMonth)}
            {renderSessionGroup("Older", groupedSessions.older)}
          </SidebarMenuSub>
        </SidebarMenu>{" "}
        {filteredSessions.length === 0 && searchQuery && (
          <div className="px-2 py-4 text-center text-sm text-muted-foreground">
            No chats found matching &quot;{searchQuery}&quot;
          </div>
        )}
        {state.sessions.length === 0 && (
          <div className="px-2 py-4 text-center text-sm text-muted-foreground">
            No chat history yet. Start a new conversation!
          </div>
        )}
      </SidebarGroupContent>
    </SidebarGroup>
  )
}
