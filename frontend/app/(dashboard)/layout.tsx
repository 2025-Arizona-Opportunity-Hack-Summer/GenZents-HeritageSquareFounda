import type React from "react"
import { AuthGuard } from "@/components/auth/auth-guard"
import { SidebarProvider } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { SidebarInset } from "@/components/ui/sidebar"
import { Header } from "@/components/layout/header"
import { ChatProvider } from "@/hooks/chat-provider"

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthGuard>
      <ChatProvider>
        <SidebarProvider className="h-screen">
          <AppSidebar />
          <SidebarInset className="flex flex-col h-screen">
            <Header />
            <main className="flex-1 overflow-hidden">{children}</main>
          </SidebarInset>
        </SidebarProvider>
      </ChatProvider>
    </AuthGuard>
  )
}
