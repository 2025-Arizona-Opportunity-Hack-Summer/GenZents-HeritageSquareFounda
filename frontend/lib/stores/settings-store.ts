import { create } from "zustand"
import { persist } from "zustand/middleware"
import { logger } from "@/lib/utils/logger"

interface SettingsState {
  // Theme and appearance
  theme: "dark" | "light" | "system"
  fontSize: "small" | "medium" | "large"
  highContrast: boolean
  layoutDensity: string

  // General
  language: string
  showFollowUpSuggestions: boolean
  chats: any[]

  // Chat and AI model settings
  defaultModel: string
  temperature: number
  systemPrompt: string
  maxTokens: number
  customInstructions: string
  communicationStyle: "professional" | "casual" | "friendly" | "balanced" | "technical"
  responseLength: "concise" | "balanced" | "detailed" | "comprehensive"
  expertiseLevel: "beginner" | "intermediate" | "advanced" | "expert"

  // Display preferences
  showTimestamps: boolean
  showWordCount: boolean
  showModelInfo: boolean
  showAvatars: boolean

  // Privacy and data
  saveHistory: boolean
  shareAnalytics: boolean
  apiKeys: {
    openai?: string
    anthropic?: string
  }

  // Connections and integrations
  googleDriveConnected: boolean
  googleDriveEmail: string
  googleDriveAutoSync: boolean
  googleDriveSyncFrequency: "realtime" | "hourly" | "daily" | "weekly"
  shareUsageWithIntegrations: boolean
  crossPlatformSync: boolean

  // Actions
  updateSetting: <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => void
  resetSettings: () => void
  exportSettings: () => string
  importSettings: (settings: string) => void
}

const defaultSettings = {
  // Theme and appearance
  theme: "system" as const,
  fontSize: "medium" as const,
  highContrast: false,
  layoutDensity: "comfortable",

  // General
  language: "en",
  showFollowUpSuggestions: true,
  chats: [],

  // Chat and AI model settings
  defaultModel: "gpt-4",
  temperature: 0.7,
  systemPrompt: "",
  maxTokens: 2048,
  customInstructions: "",
  communicationStyle: "balanced" as const,
  responseLength: "balanced" as const,
  expertiseLevel: "intermediate" as const,

  // Display preferences
  showTimestamps: false,
  showWordCount: false,
  showModelInfo: false,
  showAvatars: true,

  // Privacy and data
  saveHistory: true,
  shareAnalytics: false,
  apiKeys: {},

  // Connections and integrations
  googleDriveConnected: false,
  googleDriveEmail: "",
  googleDriveAutoSync: false,
  googleDriveSyncFrequency: "daily" as const,
  shareUsageWithIntegrations: false,
  crossPlatformSync: true,
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      ...defaultSettings,

      updateSetting: (key, value) => {
        set((state) => ({ ...state, [key]: value }))
      },

      resetSettings: () => {
        set(defaultSettings)
      },

      exportSettings: () => {
        const settings = get()
        return JSON.stringify(settings, null, 2)
      },

      importSettings: (settingsJson: string) => {
        try {
          const settings = JSON.parse(settingsJson)
          set({ ...defaultSettings, ...settings })
        } catch (error) {
          logger.error("Failed to import settings", error as Error, { component: "settings-store" })
        }
      },
    }),
    {
      name: "settings-storage",
    }
  )
)
