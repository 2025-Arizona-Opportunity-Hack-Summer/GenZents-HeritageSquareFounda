import { create } from "zustand"
import { supabase } from "@/lib/supabase/client"
import { logger } from "@/lib/utils/logger"
import type { Database, UserPermission, UserStatus } from "@/lib/types/database"

type Profile = Database["public"]["Tables"]["profiles"]["Row"]
type ProfileInsert = Database["public"]["Tables"]["profiles"]["Insert"]
type ProfileUpdate = Database["public"]["Tables"]["profiles"]["Update"]

interface AdminState {
  users: Profile[]
  loading: boolean
  fetchUsers: () => Promise<void>
  inviteUser: (email: string, fullName: string, permissions: UserPermission[]) => Promise<{ error?: string }>
  updateUser: (id: string, updates: ProfileUpdate) => Promise<{ error?: string }>
  deleteUser: (id: string) => Promise<{ error?: string }>
  toggleUserStatus: (id: string, status: UserStatus) => Promise<{ error?: string }>
  updateUserPermissions: (id: string, permissions: UserPermission[]) => Promise<{ error?: string }>
}

export const useAdminStore = create<AdminState>((set, get) => ({
  users: [],
  loading: false,

  fetchUsers: async () => {
    try {
      set({ loading: true })

      const { data: users, error } = await supabase
        .from("profiles")
        .select("*")
        .order("created_at", { ascending: false })

      if (error) {
        logger.error("Error fetching users", error as Error, { component: "admin-store" })
        return
      }

      set({ users: users || [], loading: false })
    } catch (error) {
      logger.error("Error fetching users", error as Error, { component: "admin-store" })
      set({ loading: false })
    }
  },

  inviteUser: async (email: string, fullName: string, permissions: UserPermission[]) => {
    try {
      set({ loading: true })

      const { data, error } = await supabase.auth.admin.inviteUserByEmail(email, {
        data: {
          full_name: fullName,
          permissions,
          status: "pending_invitation" as UserStatus,
          is_admin: false,
        },
        redirectTo: `${window.location.origin}/auth/set-password`,
      })

      if (error) {
        logger.error("Error inviting user", error as Error, { component: "admin-store" })
        set({ loading: false })
        return { error: error.message }
      }

      await get().fetchUsers()
      set({ loading: false })
      return {}
    } catch (error) {
      logger.error("Error inviting user", error as Error, { component: "admin-store" })
      set({ loading: false })
      return { error: "An unexpected error occurred" }
    }
  },

  updateUser: async (id: string, updates: ProfileUpdate) => {
    try {
      set({ loading: true })

      const { error } = await supabase
        .from("profiles")
        .update({ ...updates, updated_at: new Date().toISOString() })
        .eq("id", id)

      if (error) {
        logger.error("Error updating user", error as Error, { component: "admin-store" })
        set({ loading: false })
        return { error: error.message }
      }

      await get().fetchUsers()
      set({ loading: false })
      return {}
    } catch (error) {
      logger.error("Error updating user", error as Error, { component: "admin-store" })
      set({ loading: false })
      return { error: "An unexpected error occurred" }
    }
  },

  deleteUser: async (id: string) => {
    try {
      set({ loading: true })

      const { error: authError } = await supabase.auth.admin.deleteUser(id)
      if (authError) {
        logger.error("Error deleting user from auth", authError as Error, { component: "admin-store" })
      }

      const { error } = await supabase.from("profiles").delete().eq("id", id)

      if (error) {
        logger.error("Error deleting user profile", error as Error, { component: "admin-store" })
        set({ loading: false })
        return { error: error.message }
      }

      await get().fetchUsers()
      set({ loading: false })
      return {}
    } catch (error) {
      logger.error("Error deleting user", error as Error, { component: "admin-store" })
      set({ loading: false })
      return { error: "An unexpected error occurred" }
    }
  },

  toggleUserStatus: async (id: string, status: UserStatus) => {
    try {
      return await get().updateUser(id, { status })
    } catch (error) {
      logger.error("Error toggling user status", error as Error, { component: "admin-store" })
      return { error: "An unexpected error occurred" }
    }
  },

  updateUserPermissions: async (id: string, permissions: UserPermission[]) => {
    try {
      return await get().updateUser(id, { permissions })
    } catch (error) {
      logger.error("Error updating user permissions", error as Error, { component: "admin-store" })
      return { error: "An unexpected error occurred" }
    }
  },
}))
