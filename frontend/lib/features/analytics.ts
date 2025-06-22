import { supabase } from "@/lib/supabase/client"
import type { Analytics, AnalyticsQuery, AnalyticsResult } from "@/lib/types"

// Remove duplicate interfaces - now using centralized types

export class AnalyticsEngine {
  static async track(event: string, properties: Record<string, any> = {}) {
    const userId = (await supabase.auth.getUser()).data.user?.id
    const sessionId = sessionStorage.getItem("session_id") || this.generateSessionId()

    if (!sessionStorage.getItem("session_id")) {
      sessionStorage.setItem("session_id", sessionId)
    }

    const analyticsData: Partial<Analytics> = {
      user_id: userId,
      event,
      properties: {
        ...properties,
        url: window.location.href,
        referrer: document.referrer,
        user_agent: navigator.userAgent,
        timestamp: new Date().toISOString(),
      },
      session_id: sessionId,
      timestamp: new Date().toISOString(),
    }

    await supabase.from("analytics").insert(analyticsData)
  }

  static async query(query: AnalyticsQuery): Promise<AnalyticsResult> {
    const { timeRange, metrics, groupBy, filters } = query

    let baseQuery = supabase
      .from("analytics")
      .select("*")
      .gte("timestamp", timeRange.start)
      .lte("timestamp", timeRange.end)

    // Apply filters
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (Array.isArray(value)) {
          baseQuery = baseQuery.in(key, value)
        } else {
          baseQuery = baseQuery.eq(key, value)
        }
      })
    }

    const { data, error } = await baseQuery

    if (error) {
      throw new Error(`Analytics query failed: ${error.message}`)
    }

    // Process data based on metrics and groupBy
    const processedData = this.processAnalyticsData(data || [], metrics, groupBy)
    const summary = this.generateSummary(data || [])

    return {
      data: processedData,
      summary,
    }
  }

  private static processAnalyticsData(data: Analytics[], metrics: string[], groupBy?: string[]): any[] {
    if (!groupBy || groupBy.length === 0) {
      return data
    }

    const grouped = data.reduce(
      (acc, item) => {
        const key = groupBy.map((field) => item[field as keyof Analytics]).join("|")
        if (!acc[key]) {
          acc[key] = []
        }
        acc[key].push(item)
        return acc
      },
      {} as Record<string, Analytics[]>
    )

    return Object.entries(grouped).map(([key, items]) => {
      const groupValues = key.split("|")
      const result: any = {}

      groupBy.forEach((field, index) => {
        result[field] = groupValues[index]
      })

      metrics.forEach((metric) => {
        switch (metric) {
          case "count":
            result.count = items.length
            break
          case "unique_users":
            result.unique_users = new Set(items.map((i) => i.user_id)).size
            break
          case "avg_session_duration":
            // Calculate average session duration
            break
          default:
            result[metric] = items.length
        }
      })

      return result
    })
  }

  private static generateSummary(data: Analytics[]) {
    const totalEvents = data.length
    const uniqueUsers = new Set(data.map((d) => d.user_id)).size

    // Calculate top events
    const eventCounts = data.reduce(
      (acc, item) => {
        acc[item.event] = (acc[item.event] || 0) + 1
        return acc
      },
      {} as Record<string, number>
    )

    const topEvents = Object.entries(eventCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([event, count]) => ({ event, count }))

    return {
      totalEvents,
      uniqueUsers,
      averageSessionDuration: 0, // Calculate based on session data
      topEvents,
    }
  }

  private static generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  static async getUserJourney(userId: string, timeRange: { start: string; end: string }) {
    const { data, error } = await supabase
      .from("analytics")
      .select("*")
      .eq("user_id", userId)
      .gte("timestamp", timeRange.start)
      .lte("timestamp", timeRange.end)
      .order("timestamp", { ascending: true })

    if (error) {
      throw new Error(`Failed to get user journey: ${error.message}`)
    }

    return data || []
  }

  static async getFunnelAnalysis(events: string[], timeRange: { start: string; end: string }) {
    const results = []

    for (let i = 0; i < events.length; i++) {
      const event = events[i]
      const { data, error } = await supabase
        .from("analytics")
        .select("user_id")
        .eq("event", event)
        .gte("timestamp", timeRange.start)
        .lte("timestamp", timeRange.end)

      if (error) continue

      const uniqueUsers = new Set(data.map((d) => d.user_id)).size
      const conversionRate: number = i === 0 ? 100 : (uniqueUsers / results[0].users) * 100

      results.push({
        step: i + 1,
        event,
        users: uniqueUsers,
        conversionRate,
      })
    }

    return results
  }

  static async getCohortAnalysis(timeRange: { start: string; end: string }) {
    // Implementation for cohort analysis
    // This would analyze user retention over time
    return []
  }

  static async getRealtimeMetrics() {
    const now = new Date()
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000)

    const { data, error } = await supabase.from("analytics").select("*").gte("timestamp", oneHourAgo.toISOString())

    if (error) return null

    const activeUsers = new Set(data.map((d) => d.user_id)).size
    const eventsPerMinute = data.length / 60
    const topPages = this.getTopPages(data)

    return {
      activeUsers,
      eventsPerMinute,
      topPages,
      lastUpdated: new Date().toISOString(),
    }
  }

  private static getTopPages(data: Analytics[]) {
    const pageCounts = data.reduce(
      (acc, item) => {
        const url = item.properties?.url
        if (url) {
          acc[url] = (acc[url] || 0) + 1
        }
        return acc
      },
      {} as Record<string, number>
    )

    return Object.entries(pageCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([url, count]) => ({ url, count }))
  }
}
