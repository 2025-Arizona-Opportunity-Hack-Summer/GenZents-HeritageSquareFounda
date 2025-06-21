"use client"

import type React from "react"
import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuthStore } from "@/lib/stores/auth-store"
import { Loader2 } from "lucide-react"
import type { UserPermission } from "@/lib/types/database"

interface AuthGuardProps {
  children: React.ReactNode
  requireAuth?: boolean
  requireAdmin?: boolean
  requiredPermissions?: UserPermission[]
  fallback?: React.ReactNode
}

export function AuthGuard({
  children,
  requireAuth = true,
  requireAdmin = false,
  requiredPermissions = [],
  fallback,
}: AuthGuardProps): React.ReactElement | null {
  const router = useRouter()
  const { user, profile, loading, initialized, initialize, isAdmin, hasPermission } = useAuthStore()

  useEffect(() => {
    if (!initialized) {
      initialize()
    }
  }, [initialized, initialize])

  useEffect(() => {
    if (!initialized || loading) return

    if (requireAuth && !user) {
      router.push("/auth")
      return
    }

    if (requireAdmin && !isAdmin()) {
      router.push("/")
      return
    }

    if (requiredPermissions.length > 0) {
      const hasAllPermissions = requiredPermissions.every((permission) => hasPermission(permission))

      if (!hasAllPermissions) {
        router.push("/")
        return
      }
    }
  }, [
    user,
    profile,
    initialized,
    loading,
    requireAuth,
    requireAdmin,
    requiredPermissions,
    router,
    isAdmin,
    hasPermission,
  ])

  if (!initialized || loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  // TODO: Using <></> instead of fallback||null . Fix later

  if (requireAuth && !user) {
    return <></>
  }

  if (requireAdmin && !isAdmin()) {
    return <></>
  }

  if (requiredPermissions.length > 0) {
    const hasAllPermissions = requiredPermissions.every((permission) => hasPermission(permission))

    if (!hasAllPermissions) {
      return <></>
    }
  }

  return <>{children}</>
}
