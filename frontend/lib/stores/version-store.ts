import { create } from "zustand"
import { createClient } from "@/lib/supabase/client"
import { logger } from "@/lib/utils/logger"
import type { Chat } from "@/lib/types/chat"
import type { Version, VersionDiff } from "@/lib/types/version"

// Remove duplicate interfaces as they now come from centralized types

interface VersionState {
  versions: Version[]
  currentVersion: Version | null
  branches: string[]
  loading: boolean

  // Version management
  loadVersions: (chatId: string) => Promise<void>
  createVersion: (chatId: string, description?: string, tags?: string[]) => Promise<void>
  rollbackToVersion: (versionId: string) => Promise<void>
  deleteVersion: (versionId: string) => Promise<void>

  // Branching
  createBranch: (versionId: string, branchName: string) => Promise<void>
  mergeBranch: (sourceBranch: string, targetBranch: string) => Promise<void>

  // Comparison
  compareVersions: (version1Id: string, version2Id: string) => Promise<VersionDiff[]>
  getVersionDiff: (versionId: string) => Promise<VersionDiff[]>

  // Export/Import
  exportVersion: (versionId: string) => Promise<string>
  importVersion: (data: string) => Promise<void>
}

export const useVersionStore = create<VersionState>((set, get) => ({
  versions: [],
  currentVersion: null,
  branches: ["main"],
  loading: false,

  loadVersions: async (chatId: string) => {
    try {
      set({ loading: true })
      const supabase = createClient()

      const { data: versions, error } = await supabase
        .from("chat_versions")
        .select("*")
        .eq("chat_id", chatId)
        .order("created_at", { ascending: false })

      if (error) throw error

      set({ versions: versions || [], loading: false })
    } catch (error) {
      logger.error("Error loading versions", error as Error, { component: "version-store", chatId })
      set({ loading: false })
    }
  },

  createVersion: async (chatId: string, description?: string, tags: string[] = []) => {
    try {
      const supabase = createClient()

      // Get current chat data
      const { data: chat, error: chatError } = await supabase
        .from("chats")
        .select(
          `
          *,
          messages (*)
        `
        )
        .eq("id", chatId)
        .single()

      if (chatError) throw chatError

      // Get current version number
      const { data: lastVersion } = await supabase
        .from("chat_versions")
        .select("version")
        .eq("chat_id", chatId)
        .order("version", { ascending: false })
        .limit(1)
        .single()

      const nextVersion = (lastVersion?.version || 0) + 1

      // Create new version
      const { data: version, error } = await supabase
        .from("chat_versions")
        .insert({
          chat_id: chatId,
          version: nextVersion,
          title: `Version ${nextVersion}`,
          description,
          data: chat,
          tags,
          branch: "main",
        })
        .select()
        .single()

      if (error) throw error

      set((state) => ({
        versions: [version, ...state.versions],
      }))
    } catch (error) {
      logger.error("Error creating version", error as Error, { component: "version-store", chatId, name })
      throw error
    }
  },

  rollbackToVersion: async (versionId: string) => {
    try {
      const supabase = createClient()

      // Get version data
      const { data: version, error: versionError } = await supabase
        .from("chat_versions")
        .select("*")
        .eq("id", versionId)
        .single()

      if (versionError) throw versionError

      const chatData = version.data as Chat

      // Update chat with version data
      const { error: updateError } = await supabase
        .from("chats")
        .update({
          title: chatData.title,
          model: chatData.model,
          system_prompt: chatData.system_prompt,
          tags: chatData.tags,
          updated_at: new Date().toISOString(),
        })
        .eq("id", chatData.id)

      if (updateError) throw updateError

      // Replace messages
      await supabase.from("messages").delete().eq("chat_id", chatData.id)

      if (chatData.messages && chatData.messages.length > 0) {
        await supabase.from("messages").insert(chatData.messages)
      }

      // Create a new version for the rollback
      await get().createVersion(chatData.id, `Rolled back to version ${version.version}`, ["rollback"])
    } catch (error) {
      logger.error("Error rolling back to version", error as Error, { component: "version-store", versionId })
      throw error
    }
  },

  deleteVersion: async (versionId: string) => {
    try {
      const supabase = createClient()

      const { error } = await supabase.from("chat_versions").delete().eq("id", versionId)

      if (error) throw error

      set((state) => ({
        versions: state.versions.filter((v) => v.id !== versionId),
      }))
    } catch (error) {
      logger.error("Error deleting version", error as Error, { component: "version-store", versionId })
      throw error
    }
  },

  createBranch: async (versionId: string, branchName: string) => {
    try {
      const supabase = createClient()

      // Get version data
      const { data: version, error } = await supabase.from("chat_versions").select("*").eq("id", versionId).single()

      if (error) throw error

      // Create new version with branch
      const { error: createError } = await supabase.from("chat_versions").insert({
        ...version,
        id: undefined,
        branch: branchName,
        parent_version: versionId,
        created_at: new Date().toISOString(),
      })

      if (createError) throw createError

      set((state) => ({
        branches: [...state.branches, branchName],
      }))
    } catch (error) {
      logger.error("Error creating branch", error as Error, { component: "version-store", versionId, branchName })
      throw error
    }
  },

  mergeBranch: async (sourceBranch: string, targetBranch: string) => {
    try {
      // Placeholder for branch merging logic
      logger.info(`Merging ${sourceBranch} into ${targetBranch}`, { component: "version-store" })

      // In a real implementation, this would:
      // 1. Compare the branches
      // 2. Resolve conflicts
      // 3. Create a merge commit
      // 4. Update the target branch
    } catch (error) {
      logger.error("Error merging branch", error as Error, { component: "version-store", sourceBranch, targetBranch })
      throw error
    }
  },

  compareVersions: async (version1Id: string, version2Id: string) => {
    try {
      const supabase = createClient()

      const [{ data: version1 }, { data: version2 }] = await Promise.all([
        supabase.from("chat_versions").select("*").eq("id", version1Id).single(),
        supabase.from("chat_versions").select("*").eq("id", version2Id).single(),
      ])

      if (!version1 || !version2) throw new Error("Versions not found")

      // Simple diff implementation (would be more sophisticated in production)
      const diffs: VersionDiff[] = []

      const data1 = version1.data as Chat
      const data2 = version2.data as Chat

      if (data1.title !== data2.title) {
        diffs.push({
          type: "modified",
          path: "title",
          oldValue: data1.title,
          newValue: data2.title,
        })
      }

      // Compare messages
      const messages1 = data1.messages || []
      const messages2 = data2.messages || []

      if (messages1.length !== messages2.length) {
        diffs.push({
          type: messages1.length > messages2.length ? "deleted" : "added",
          path: "messages",
          oldValue: messages1.length,
          newValue: messages2.length,
        })
      }

      return diffs
    } catch (error) {
      logger.error("Error comparing versions", error as Error, { component: "version-store", version1Id, version2Id })
      return []
    }
  },

  getVersionDiff: async (versionId: string) => {
    try {
      const version = get().versions.find((v) => v.id === versionId)
      if (!version || !version.parentVersion) return []

      return await get().compareVersions(version.parentVersion, versionId)
    } catch (error) {
      logger.error("Error getting version diff", error as Error, { component: "version-store", versionId })
      return []
    }
  },

  exportVersion: async (versionId: string) => {
    try {
      const version = get().versions.find((v) => v.id === versionId)
      if (!version) throw new Error("Version not found")

      return JSON.stringify(version, null, 2)
    } catch (error) {
      logger.error("Error exporting version", error as Error, { component: "version-store", versionId })
      throw error
    }
  },

  importVersion: async (data: string) => {
    try {
      const version = JSON.parse(data)
      const supabase = createClient()

      const { error } = await supabase.from("chat_versions").insert({
        ...version,
        id: undefined,
        created_at: new Date().toISOString(),
      })

      if (error) throw error

      await get().loadVersions(version.chatId)
    } catch (error) {
      logger.error("Error importing version", error as Error, { component: "version-store" })
      throw error
    }
  },
}))
