import { create } from "zustand"
import { supabase } from "@/lib/supabase/client"
import { logger } from "@/lib/utils/logger"
import type { User, Session } from "@supabase/supabase-js"
import type { Database, UserPermission, UserStatus } from "@/lib/types/database"

type Profile = Database["public"]["Tables"]["profiles"]["Row"]

interface AuthState {
  user: User | null
  profile: Profile | null
  session: Session | null
  loading: boolean
  initialized: boolean
  signIn: (email: string, password: string) => Promise<{ error?: string }>
  signOut: () => Promise<void>
  initialize: () => Promise<void>
  updateProfile: (updates: Partial<Profile>) => Promise<{ error?: string }>
  resetPassword: (email: string) => Promise<{ error?: string }>
  updatePassword: (password: string) => Promise<{ error?: string }>
  setPassword: (password: string, token?: string) => Promise<{ error?: string }>
  isAdmin: () => boolean
  hasPermission: (permission: UserPermission) => boolean
  refreshProfile: () => Promise<void>
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  profile: null,
  session: null,
  loading: false,
  initialized: false,

  initialize: async () => {
    try {
      set({ loading: true })

      const {
        data: { session },
        error,
      } = await supabase.auth.getSession()

      if (error) {
        logger.error("Error getting session", error as Error, { component: "auth-store" })
      }

      if (session?.user) {
        await get().refreshProfile()
      }

      set({
        session,
        user: session?.user || null,
        loading: false,
        initialized: true,
      })

      supabase.auth.onAuthStateChange(async (event, session) => {
        if (event === "SIGNED_IN" && session?.user) {
          await get().refreshProfile()
        } else if (event === "SIGNED_OUT") {
          set({ profile: null })
        }

        set({
          session,
          user: session?.user || null,
        })
      })
    } catch (error) {
      logger.error("Error initializing auth", error as Error, { component: "auth-store" })
      set({ loading: false, initialized: true })
    }
  },

  refreshProfile: async () => {
    const { user } = get()
    if (!user) return

    try {
      const { data: profile, error } = await supabase.from("profiles").select("*").eq("id", user.id).single()

      if (error) {
        logger.error("Error fetching profile", error as Error, { component: "auth-store" })
        return
      }

      set({ profile })
    } catch (error) {
      logger.error("Error refreshing profile", error as Error, { component: "auth-store" })
    }
  },

  signIn: async (email: string, password: string) => {
    try {
      set({ loading: true })
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })

      if (error) {
        set({ loading: false })
        return { error: error.message }
      }

      await supabase.from("profiles").update({ last_sign_in: new Date().toISOString() }).eq("id", data.user.id)

      await get().refreshProfile()
      set({ loading: false })
      return {}
    } catch (error) {
      logger.error("Error signing in", error as Error, { component: "auth-store" })
      set({ loading: false })
      return { error: "An unexpected error occurred" }
    }
  },

  signOut: async () => {
    try {
      const { error } = await supabase.auth.signOut()
      if (error) {
        logger.error("Error signing out", error as Error, { component: "auth-store" })
      }
      set({ user: null, profile: null, session: null })
    } catch (error) {
      logger.error("Error signing out", error as Error, { component: "auth-store" })
    }
  },

  updateProfile: async (updates: Partial<Profile>) => {
    try {
      const { user } = get()
      if (!user) return { error: "No authenticated user" }

      const { error } = await supabase
        .from("profiles")
        .update({ ...updates, updated_at: new Date().toISOString() })
        .eq("id", user.id)

      if (error) {
        logger.error("Error updating profile", error as Error, { component: "auth-store" })
        return { error: error.message }
      }

      await get().refreshProfile()
      return {}
    } catch (error) {
      logger.error("Error updating profile", error as Error, { component: "auth-store" })
      return { error: "An unexpected error occurred" }
    }
  },

  resetPassword: async (email: string) => {
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/auth/reset-password`,
      })

      if (error) {
        return { error: error.message }
      }

      return {}
    } catch (error) {
      logger.error("Error resetting password", error as Error, { component: "auth-store" })
      return { error: "An unexpected error occurred" }
    }
  },

  updatePassword: async (password: string) => {
    try {
      const { error } = await supabase.auth.updateUser({ password })

      if (error) {
        return { error: error.message }
      }

      return {}
    } catch (error) {
      logger.error("Error updating password", error as Error, { component: "auth-store" })
      return { error: "An unexpected error occurred" }
    }
  },

  setPassword: async (password: string, token?: string) => {
    try {
      if (token) {
        const { error } = await supabase.auth.verifyOtp({
          token_hash: token,
          type: "recovery",
        })

        if (error) {
          return { error: error.message }
        }
      }

      const { error } = await supabase.auth.updateUser({ password })

      if (error) {
        return { error: error.message }
      }

      return {}
    } catch (error) {
      logger.error("Error setting password", error as Error, { component: "auth-store" })
      return { error: "An unexpected error occurred" }
    }
  },

  isAdmin: () => {
    const { profile } = get()
    return profile?.is_admin === true
  },

  hasPermission: (permission: UserPermission) => {
    const { profile } = get()
    if (!profile) return false
    if (profile.is_admin) return true
    return profile.permissions?.includes(permission) === true
  },
}))
