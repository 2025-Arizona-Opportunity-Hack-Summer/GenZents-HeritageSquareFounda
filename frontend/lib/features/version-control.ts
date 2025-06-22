import { supabase } from "@/lib/supabase/client"
import type { ChatVersion, VersionChange } from "@/lib/types"

export class VersionControl {
  static async createVersion(
    chatId: string,
    title: string,
    description: string,
    changes: VersionChange[]
  ): Promise<string> {
    const { data: currentVersions } = await supabase
      .from("chat_versions")
      .select("version")
      .eq("chat_id", chatId)
      .order("version", { ascending: false })
      .limit(1)

    const nextVersion = (currentVersions?.[0]?.version || 0) + 1

    const { data: version, error } = await supabase
      .from("chat_versions")
      .insert({
        chat_id: chatId,
        version: nextVersion,
        title,
        description,
        changes,
        created_by: (await supabase.auth.getUser()).data.user?.id,
        created_at: new Date().toISOString(),
        branch: "main",
      })
      .select()
      .single()

    if (error) throw error
    return version.id
  }

  static async getVersionHistory(chatId: string): Promise<ChatVersion[]> {
    const { data, error } = await supabase
      .from("chat_versions")
      .select(
        `
        *,
        created_by_user:users!created_by(id, name, email)
      `
      )
      .eq("chat_id", chatId)
      .order("version", { ascending: false })

    if (error) throw error
    return data || []
  }

  static async rollbackToVersion(chatId: string, versionId: string): Promise<void> {
    const { data: version, error: versionError } = await supabase
      .from("chat_versions")
      .select("*")
      .eq("id", versionId)
      .single()

    if (versionError || !version) {
      throw new Error("Version not found")
    }

    // Get the chat state at this version
    const { data: versionData, error: dataError } = await supabase
      .from("chat_version_data")
      .select("data")
      .eq("version_id", versionId)
      .single()

    if (dataError || !versionData) {
      throw new Error("Version data not found")
    }

    // Restore the chat to this state
    const chatData = JSON.parse(versionData.data)

    await supabase.from("chats").update(chatData.chat).eq("id", chatId)

    // Restore messages
    await supabase.from("messages").delete().eq("chat_id", chatId)

    if (chatData.messages?.length > 0) {
      await supabase.from("messages").insert(chatData.messages)
    }

    // Create a new version for this rollback
    await this.createVersion(
      chatId,
      `Rollback to version ${version.version}`,
      `Rolled back to version ${version.version}: ${version.title}`,
      [
        {
          type: "modified",
          resource: "chat",
          resource_id: chatId,
          old_value: null,
          new_value: chatData,
        },
      ]
    )
  }

  static async createBranch(chatId: string, branchName: string, fromVersion?: string): Promise<string> {
    const sourceVersion = fromVersion || (await this.getLatestVersion(chatId))

    const { data: branch, error } = await supabase
      .from("chat_branches")
      .insert({
        chat_id: chatId,
        name: branchName,
        created_from: sourceVersion,
        created_by: (await supabase.auth.getUser()).data.user?.id,
        created_at: new Date().toISOString(),
      })
      .select()
      .single()

    if (error) throw error
    return branch.id
  }

  static async mergeBranch(chatId: string, sourceBranch: string, targetBranch = "main"): Promise<void> {
    // Get changes from source branch
    const { data: sourceChanges } = await supabase
      .from("chat_versions")
      .select("changes")
      .eq("chat_id", chatId)
      .eq("branch", sourceBranch)
      .order("version", { ascending: true })

    if (!sourceChanges?.length) {
      throw new Error("No changes to merge")
    }

    // Apply changes to target branch
    const allChanges = sourceChanges.flatMap((v) => v.changes)

    await this.createVersion(
      chatId,
      `Merge ${sourceBranch} into ${targetBranch}`,
      `Merged changes from ${sourceBranch} branch`,
      allChanges
    )
  }

  static async compareVersions(chatId: string, version1: string, version2: string): Promise<VersionChange[]> {
    const [v1Data, v2Data] = await Promise.all([this.getVersionData(version1), this.getVersionData(version2)])

    return this.calculateDiff(v1Data, v2Data)
  }

  private static async getLatestVersion(chatId: string): Promise<string> {
    const { data } = await supabase
      .from("chat_versions")
      .select("id")
      .eq("chat_id", chatId)
      .order("version", { ascending: false })
      .limit(1)
      .single()

    return data?.id || ""
  }

  private static async getVersionData(versionId: string): Promise<any> {
    const { data } = await supabase.from("chat_version_data").select("data").eq("version_id", versionId).single()

    return data ? JSON.parse(data.data) : null
  }

  private static calculateDiff(data1: any, data2: any): VersionChange[] {
    // Implementation for calculating differences between versions
    const changes: VersionChange[] = []

    // Compare chat properties
    Object.keys(data2.chat || {}).forEach((key) => {
      if (data1.chat?.[key] !== data2.chat?.[key]) {
        changes.push({
          type: "modified",
          resource: "chat",
          resource_id: data2.chat.id,
          old_value: data1.chat?.[key],
          new_value: data2.chat?.[key],
        })
      }
    })

    // Compare messages
    const messages1 = data1.messages || []
    const messages2 = data2.messages || []

    messages2.forEach((msg2: any) => {
      const msg1 = messages1.find((m: any) => m.id === msg2.id)
      if (!msg1) {
        changes.push({
          type: "added",
          resource: "message",
          resource_id: msg2.id,
          old_value: null,
          new_value: msg2,
        })
      } else if (JSON.stringify(msg1) !== JSON.stringify(msg2)) {
        changes.push({
          type: "modified",
          resource: "message",
          resource_id: msg2.id,
          old_value: msg1,
          new_value: msg2,
        })
      }
    })

    messages1.forEach((msg1: any) => {
      const msg2 = messages2.find((m: any) => m.id === msg1.id)
      if (!msg2) {
        changes.push({
          type: "deleted",
          resource: "message",
          resource_id: msg1.id,
          old_value: msg1,
          new_value: null,
        })
      }
    })

    return changes
  }
}
