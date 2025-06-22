import { create } from "zustand"
import { createClient } from "@/lib/supabase/client"
import { logger } from "@/lib/utils/logger"

export interface UsageMetrics {
  totalChats: number
  totalMessages: number
  totalTokens: number
  totalCost: number
  averageResponseTime: number
  mostUsedModel: string
  dailyUsage: Array<{ date: string; messages: number; tokens: number }>
  modelUsage: Array<{ model: string; count: number; percentage: number }>
  topicDistribution: Array<{ topic: string; count: number }>
}

export interface PerformanceMetrics {
  responseTime: number
  throughput: number
  errorRate: number
  uptime: number
  memoryUsage: number
  cpuUsage: number
  networkLatency: number
}

export interface UserActivity {
  activeUsers: number
  newUsers: number
  returningUsers: number
  sessionDuration: number
  bounceRate: number
  engagementRate: number
}

interface AnalyticsState {
  usageMetrics: UsageMetrics | null
  performanceMetrics: PerformanceMetrics | null
  userActivity: UserActivity | null
  loading: boolean
  dateRange: { start: Date; end: Date }

  // Actions
  loadUsageMetrics: () => Promise<void>
  loadPerformanceMetrics: () => Promise<void>
  loadUserActivity: () => Promise<void>
  setDateRange: (start: Date, end: Date) => void
  exportAnalytics: (format: "csv" | "json" | "pdf") => Promise<string>

  // Real-time monitoring
  startRealTimeMonitoring: () => void
  stopRealTimeMonitoring: () => void
}

export const useAnalyticsStore = create<AnalyticsState>((set, get) => ({
  usageMetrics: null,
  performanceMetrics: null,
  userActivity: null,
  loading: false,
  dateRange: {
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
    end: new Date(),
  },

  loadUsageMetrics: async () => {
    try {
      set({ loading: true })
      const supabase = createClient()

      // Get basic counts
      const [{ count: totalChats }, { count: totalMessages }] = await Promise.all([
        supabase.from("chats").select("*", { count: "exact", head: true }),
        supabase.from("messages").select("*", { count: "exact", head: true }),
      ])

      // Get daily usage (mock data for now)
      const dailyUsage = Array.from({ length: 30 }, (_, i) => {
        const date = new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000)
        return {
          date: date.toISOString().split("T")[0],
          messages: Math.floor(Math.random() * 100) + 10,
          tokens: Math.floor(Math.random() * 10000) + 1000,
        }
      })

      // Mock model usage data
      const modelUsage = [
        { model: "gpt-4", count: 150, percentage: 60 },
        { model: "gpt-3.5-turbo", count: 75, percentage: 30 },
        { model: "claude-3", count: 25, percentage: 10 },
      ]

      // Mock topic distribution
      const topicDistribution = [
        { topic: "Programming", count: 45 },
        { topic: "Writing", count: 30 },
        { topic: "Analysis", count: 25 },
        { topic: "Creative", count: 20 },
        { topic: "Research", count: 15 },
      ]

      const usageMetrics: UsageMetrics = {
        totalChats: totalChats || 0,
        totalMessages: totalMessages || 0,
        totalTokens: dailyUsage.reduce((sum, day) => sum + day.tokens, 0),
        totalCost: 45.67, // Mock cost
        averageResponseTime: 1.2, // Mock response time
        mostUsedModel: "gpt-4",
        dailyUsage,
        modelUsage,
        topicDistribution,
      }

      set({ usageMetrics, loading: false })
    } catch (error) {
      logger.error("Error loading usage metrics", error as Error, { component: "analytics-store" })
      set({ loading: false })
    }
  },

  loadPerformanceMetrics: async () => {
    try {
      // Mock performance data (would be real metrics in production)
      const performanceMetrics: PerformanceMetrics = {
        responseTime: 1200 + Math.random() * 300, // ms
        throughput: 45 + Math.random() * 10, // requests/min
        errorRate: Math.random() * 2, // percentage
        uptime: 99.8 + Math.random() * 0.2, // percentage
        memoryUsage: 65 + Math.random() * 20, // percentage
        cpuUsage: 35 + Math.random() * 30, // percentage
        networkLatency: 50 + Math.random() * 20, // ms
      }

      set({ performanceMetrics })
    } catch (error) {
      logger.error("Error loading performance metrics", error as Error, { component: "analytics-store" })
    }
  },

  loadUserActivity: async () => {
    try {
      // Mock user activity data
      const userActivity: UserActivity = {
        activeUsers: 1247,
        newUsers: 89,
        returningUsers: 1158,
        sessionDuration: 18.5, // minutes
        bounceRate: 12.3, // percentage
        engagementRate: 87.7, // percentage
      }

      set({ userActivity })
    } catch (error) {
      logger.error("Error loading user activity", error as Error, { component: "analytics-store" })
    }
  },

  setDateRange: (start: Date, end: Date) => {
    set({ dateRange: { start, end } })
    // Reload metrics with new date range
    get().loadUsageMetrics()
    get().loadUserActivity()
  },

  exportAnalytics: async (format: "csv" | "json" | "pdf") => {
    try {
      const { usageMetrics, performanceMetrics, userActivity } = get()

      const data = {
        usageMetrics,
        performanceMetrics,
        userActivity,
        exportedAt: new Date().toISOString(),
      }

      switch (format) {
        case "json":
          return JSON.stringify(data, null, 2)

        case "csv":
          // Simple CSV export (would be more sophisticated in production)
          let csv = "Metric,Value\n"
          if (usageMetrics) {
            csv += `Total Chats,${usageMetrics.totalChats}\n`
            csv += `Total Messages,${usageMetrics.totalMessages}\n`
            csv += `Total Tokens,${usageMetrics.totalTokens}\n`
            csv += `Total Cost,$${usageMetrics.totalCost}\n`
          }
          return csv

        case "pdf":
          // PDF export would require a PDF library
          return "PDF export not implemented yet"

        default:
          throw new Error("Unsupported format")
      }
    } catch (error) {
      logger.error("Error exporting analytics", error as Error, { component: "analytics-store" })
      throw error
    }
  },

  startRealTimeMonitoring: () => {
    // Start real-time monitoring (would use WebSocket or Server-Sent Events)
    logger.info("Starting real-time monitoring", { component: "analytics-store" })

    const interval = setInterval(() => {
      get().loadPerformanceMetrics()
    }, 5000) // Update every 5 seconds

    // Store interval ID for cleanup
    ;(window as any).monitoringInterval = interval
  },

  stopRealTimeMonitoring: () => {
    logger.info("Stopping real-time monitoring", { component: "analytics-store" })

    if ((window as any).monitoringInterval) {
      clearInterval((window as any).monitoringInterval)
      delete (window as any).monitoringInterval
    }
  },
}))
