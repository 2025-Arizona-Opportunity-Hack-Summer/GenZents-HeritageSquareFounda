import { supabase } from "@/lib/supabase/client"
import type { Chat, Message, SearchOptions, SearchResult, SearchFacets } from "@/lib/types"

// Remove duplicate interfaces - now using centralized types

export class AdvancedSearch {
  static async search(options: SearchOptions): Promise<SearchResult> {
    const { query, filters, sortBy = "relevance", sortOrder = "desc", limit = 50, offset = 0 } = options

    // Build search query
    let chatQuery = supabase.from("chats").select("*")
    let messageQuery = supabase.from("messages").select("*")

    // Apply text search
    if (query) {
      chatQuery = chatQuery.textSearch("title,description", query)
      messageQuery = messageQuery.textSearch("content", query)
    }

    // Apply filters
    if (filters.dateRange) {
      const [start, end] = filters.dateRange
      chatQuery = chatQuery.gte("created_at", start).lte("created_at", end)
      messageQuery = messageQuery.gte("created_at", start).lte("created_at", end)
    }

    if (filters.users?.length) {
      chatQuery = chatQuery.in("user_id", filters.users)
      messageQuery = messageQuery.in("user_id", filters.users)
    }

    if (filters.tags?.length) {
      chatQuery = chatQuery.overlaps("tags", filters.tags)
    }

    if (filters.models?.length) {
      messageQuery = messageQuery.in("metadata->model", filters.models)
    }

    if (filters.hasAttachments !== undefined) {
      if (filters.hasAttachments) {
        messageQuery = messageQuery.not("attachments", "is", null)
      } else {
        messageQuery = messageQuery.is("attachments", null)
      }
    }

    if (filters.isEdited !== undefined) {
      messageQuery = messageQuery.eq("edited", filters.isEdited)
    }

    if (filters.minTokens) {
      messageQuery = messageQuery.gte("metadata->tokens", filters.minTokens)
    }

    if (filters.maxTokens) {
      messageQuery = messageQuery.lte("metadata->tokens", filters.maxTokens)
    }

    // Apply sorting
    const sortColumn = this.getSortColumn(sortBy)
    chatQuery = chatQuery.order(sortColumn, { ascending: sortOrder === "asc" })
    messageQuery = messageQuery.order(sortColumn, { ascending: sortOrder === "asc" })

    // Apply pagination
    chatQuery = chatQuery.range(offset, offset + limit - 1)
    messageQuery = messageQuery.range(offset, offset + limit - 1)

    // Execute queries
    const [chatResult, messageResult] = await Promise.all([chatQuery, messageQuery])

    // Get facets
    const facets = await this.getFacets(query, filters)

    // Get suggestions
    const suggestions = await this.getSuggestions(query)

    return {
      chats: chatResult.data || [],
      messages: messageResult.data || [],
      totalCount: (chatResult.data?.length || 0) + (messageResult.data?.length || 0),
      facets,
      suggestions,
    }
  }

  private static getSortColumn(sortBy: string): string {
    switch (sortBy) {
      case "date":
        return "created_at"
      case "tokens":
        return "metadata->tokens"
      case "reactions":
        return "reactions"
      default:
        return "created_at"
    }
  }

  private static async getFacets(query: string, filters: any): Promise<SearchFacets> {
    // Implementation for getting search facets
    return {
      users: [],
      tags: [],
      models: [],
      dateRanges: [],
    }
  }

  private static async getSuggestions(query: string): Promise<string[]> {
    // Implementation for getting search suggestions
    return []
  }

  static async saveSearch(userId: string, searchOptions: SearchOptions, name: string) {
    await supabase.from("saved_searches").insert({
      user_id: userId,
      name,
      options: searchOptions,
      created_at: new Date().toISOString(),
    })
  }

  static async getSavedSearches(userId: string) {
    const { data } = await supabase
      .from("saved_searches")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })

    return data || []
  }
}
