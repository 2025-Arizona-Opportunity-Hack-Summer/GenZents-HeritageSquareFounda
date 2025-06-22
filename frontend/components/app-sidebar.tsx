"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuAction,
  SidebarRail,
} from "@/components/ui/sidebar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { MessageSquare, Plus, LogOut, MoreHorizontal, User, Wrench, GitBranch, FolderOpen, Shield } from "lucide-react"
import { useChatStore } from "@/lib/stores/chat-store"
import { useAuthStore } from "@/lib/stores/auth-store"

export function AppSidebar() {
  const router = useRouter()
  const { chats, createChat, selectChat, deleteChat, loadChats, currentChatId } = useChatStore()
  const { user, profile, signOut, isAdmin, hasPermission } = useAuthStore()
  useEffect(() => {
    loadChats()
  }, [loadChats])

  const handleDeleteChat = async (chatId: string) => {
    await deleteChat(chatId)
    if (currentChatId === chatId) {
      router.push("/")
    }
  }

  const handleNewChat = async () => {
    const chatId = await createChat()
    router.push(`/chat/${chatId}`)
  }

  const handleSelectChat = (chatId: string) => {
    selectChat(chatId)
    router.push(`/chat/${chatId}`)
  }

  const handleSignOut = async () => {
    await signOut()
    router.push("/auth")
  }

  const navigationItems = [
    { title: "Tools", url: "/tools", icon: Wrench, permission: "tools_access" },
    { title: "Context", url: "/context", icon: FolderOpen, permission: "context_management" },
    { title: "Version", url: "/version", icon: GitBranch, permission: "version_history" },
  ].filter((item) => hasPermission(item.permission as any))

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="flex items-center justify-between p-2">
          <a
            href="/"
            className="text-lg font-semibold cursor-pointer hover:text-primary transition-colors"
            onClick={(e) => {
              e.preventDefault()
              router.push("/")
            }}
          >
            Archyx AI
          </a>
          <div className="flex items-center gap-1">
            <Button onClick={handleNewChat} size="sm" variant="ghost">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent className="flex flex-col overflow-hidden">
        <SidebarGroup className="flex-1 min-h-0">
          <SidebarGroupLabel>Recent Chats</SidebarGroupLabel>
          <SidebarGroupContent className="flex-1 overflow-auto">
            <SidebarMenu>
              {chats.map((chat) => (
                <SidebarMenuItem key={chat.id}>
                  <SidebarMenuButton
                    asChild
                    isActive={currentChatId === chat.id}
                  >
                    <a 
                      href={`/chat/${chat.id}`}
                      className="w-full text-left"
                      onClick={(e) => {
                        e.preventDefault()
                        handleSelectChat(chat.id)
                      }}
                    >
                      <MessageSquare className="h-4 w-4" />
                      <span className="truncate">{chat.title}</span>
                    </a>
                  </SidebarMenuButton>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <SidebarMenuAction>
                        <MoreHorizontal className="h-4 w-4" />
                      </SidebarMenuAction>
                    </DropdownMenuTrigger>{" "}
                    <DropdownMenuContent>
                      <DropdownMenuItem onClick={() => handleDeleteChat(chat.id)}>Delete</DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="mt-6 flex-shrink-0">
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="space-y-2">
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <a 
                      href={item.url}
                      className="w-full text-left"
                      onClick={(e) => {
                        e.preventDefault()
                        router.push(item.url)
                      }}
                    >
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton>
                  <Avatar className="h-6 w-6">
                    <AvatarFallback>{user?.email?.charAt(0).toUpperCase() || "U"}</AvatarFallback>
                  </Avatar>
                  <div className="flex flex-col flex-1 min-w-0">
                    <span className="truncate text-sm">{profile?.full_name || user?.email}</span>
                    {isAdmin() && (
                      <Badge variant="outline" className="text-xs w-fit">
                        Admin
                      </Badge>
                    )}
                  </div>
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="top" className="w-[--radix-popper-anchor-width]">
                <DropdownMenuItem onClick={() => router.push("/account")}>
                  <User className="h-4 w-4 mr-2" />
                  Account
                </DropdownMenuItem>
                {isAdmin() && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => router.push("/admin")}>
                      <Shield className="h-4 w-4 mr-2" />
                      Admin Panel
                    </DropdownMenuItem>
                  </>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleSignOut}>
                  <LogOut className="h-4 w-4 mr-2" />
                  Sign Out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  )
}
