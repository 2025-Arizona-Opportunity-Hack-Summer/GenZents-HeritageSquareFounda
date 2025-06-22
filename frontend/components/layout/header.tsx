"use client"

import { SidebarTrigger } from "@/components/ui/sidebar"
import { Separator } from "@/components/ui/separator"
import { ThemeToggle } from "@/components/theme-toggle"
import { Button } from "@/components/ui/button"
import { Settings, Search, Bell, HelpCircle } from "lucide-react"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Badge } from "@/components/ui/badge"
import { useChat } from "@/hooks/chat-provider"
import { SettingsDialog } from "@/components/settings/settings-dialog"
import { usePathname } from "next/navigation"
import { useChatStore } from "@/lib/stores/chat-store"

export function Header() {
  const { currentSession } = useChat()
  const pathname = usePathname()
  const { getCurrentChat } = useChatStore()

  const getPageTitle = () => {
    if (pathname === "/") {
      return "Archyx AI"
    }
    
    if (pathname.startsWith("/chat/")) {
      const currentChat = getCurrentChat()
      return currentChat?.title || "Chat"
    }
    
    switch (pathname) {
      case "/tools":
        return "Tools"
      case "/context":
        return "Context"
      case "/version":
        return "Version"
      case "/admin":
        return "Admin"
      case "/account":
        return "Account"
      default:
        // For any other paths, capitalize the first segment
        const segments = pathname.split("/").filter(Boolean)
        return segments.length > 0 
          ? segments[0].charAt(0).toUpperCase() + segments[0].slice(1)
          : "Archyx AI"
    }
  }

  return (
    <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-2 h-4" />

      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem className="hidden md:block">
            <BreadcrumbLink href="/">{getPageTitle()}</BreadcrumbLink>
          </BreadcrumbItem>
          {currentSession && pathname.startsWith("/chat/") && (
            <>
              <BreadcrumbSeparator className="hidden md:block" />
              <BreadcrumbItem>
                <BreadcrumbPage className="max-w-[200px] truncate">{currentSession.title}</BreadcrumbPage>
              </BreadcrumbItem>
            </>
          )}
        </BreadcrumbList>
      </Breadcrumb>

      <div className="ml-auto flex items-center gap-2">
        <SettingsDialog>
          <Button variant="ghost" size="sm">
            <Settings className="h-4 w-4" />
          </Button>
        </SettingsDialog>

        <ThemeToggle />
      </div>
    </header>
  )
}
