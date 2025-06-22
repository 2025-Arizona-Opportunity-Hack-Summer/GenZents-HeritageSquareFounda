"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect, type ReactNode } from "react"
import { logger } from "@/lib/utils/logger"
import { SecureStorage } from "@/lib/utils/local-storage"
import type { Settings } from "@/lib/types/user"

// Extend the centralized Settings interface with additional properties specific to this provider
interface ExtendedSettings extends Settings {
  // Privacy and data
  saveHistory: boolean
  shareAnalytics: boolean
  apiKeys: {
    openai?: string
    anthropic?: string
  }

  // Developer settings
  showDebugInfo: boolean
  enableAdvancedFeatures: boolean
  customCSS: string
  apiEndpoint: string

  // Collaboration
  allowCollaboration: boolean
  defaultPermissions: string
  invitationMode: string

  // Shortcuts and productivity
  keyboardShortcuts: Record<string, any>
  shortcuts: any[]
  autoSave: boolean
  autoSaveInterval: number

  // Connections and integrations
  googleDriveConnected: boolean
  googleDriveEmail: string
  googleDriveAutoSync: boolean
  googleDriveSyncFrequency: "realtime" | "hourly" | "daily" | "weekly"
  shareUsageWithIntegrations: boolean
  crossPlatformSync: boolean
}

interface SettingsContextProps {
  settings: ExtendedSettings
  updateSetting: (key: keyof ExtendedSettings, value: any) => void
  resetSettings: () => void
  exportSettings: () => void
  importSettings: (importedSettings: Partial<ExtendedSettings>) => void
}

const defaultSettings: ExtendedSettings = {
  // Theme and appearance
  fontSize: "medium",
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
  communicationStyle: "balanced",
  responseLength: "balanced",
  expertiseLevel: "intermediate",

  // Display preferences
  showTimestamps: false,
  showWordCount: false,
  showModelInfo: false,
  showTokenCount: false,
  compactMode: false,
  fullScreenMode: false,
  showAvatars: true,

  // Export settings
  exportFormat: "json",
  includeMetadata: true,
  includeSystemMessages: true,

  // Security and privacy
  encryptMessages: false,
  retentionDays: 30,
  allowTelemetry: false,

  // Developer settings
  showDebugInfo: false,
  enableAdvancedFeatures: false,
  customCSS: "",
  apiEndpoint: "",

  // Collaboration
  allowCollaboration: true,
  defaultPermissions: "view",
  invitationMode: "manual",

  // Shortcuts and productivity
  keyboardShortcuts: {},
  shortcuts: [],
  autoSave: true,
  autoSaveInterval: 30,

  // Privacy and data
  saveHistory: true,
  shareAnalytics: false,
  apiKeys: {},

  // Connections and integrations
  googleDriveConnected: false,
  googleDriveEmail: "",
  googleDriveAutoSync: false,
  googleDriveSyncFrequency: "daily",
  shareUsageWithIntegrations: false,
  crossPlatformSync: true,
}

const SettingsContext = createContext<SettingsContextProps>({
  settings: defaultSettings,
  updateSetting: () => {},
  resetSettings: () => {},
  exportSettings: () => {},
  importSettings: () => {},
})

interface SettingsProviderProps {
  children: ReactNode
}

export const SettingsProvider: React.FC<SettingsProviderProps> = ({ children }) => {
  const [settings, setSettings] = useState<ExtendedSettings>(() => {
    const loadSettings = async () => {
      try {
        const storedSettings = await SecureStorage.getItem<ExtendedSettings>("settings")
        return storedSettings ? { ...defaultSettings, ...storedSettings } : defaultSettings
      } catch (error) {
        logger.error("Error loading stored settings", error as Error, { component: "settings-provider" })
        return defaultSettings
      }
    }
    // For now, return default settings and load async in useEffect
    return defaultSettings
  })

  // Load settings from secure storage on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const storedSettings = await SecureStorage.getItem<ExtendedSettings>("settings")
        if (storedSettings) {
          setSettings(prev => ({ ...prev, ...storedSettings }))
        }
      } catch (error) {
        logger.error("Error loading stored settings", error as Error, { component: "settings-provider" })
      }
    }
    loadSettings()
  }, [])

  useEffect(() => {
    SecureStorage.setItem("settings", settings, { 
      encrypt: false, // Settings don't need encryption but could be enabled for sensitive data
      expiryHours: 24 * 30 // 30 days
    })
  }, [settings])

  const updateSetting = (key: keyof ExtendedSettings, value: any) => {
    setSettings((prevSettings) => ({
      ...prevSettings,
      [key]: value,
    }))
  }

  const resetSettings = () => {
    setSettings(defaultSettings)
  }

  const exportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2)
    const dataBlob = new Blob([dataStr], { type: "application/json" })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement("a")
    link.href = url
    link.download = "settings.json"
    link.click()
    URL.revokeObjectURL(url)
  }

  const importSettings = (importedSettings: Partial<ExtendedSettings>) => {
    setSettings((prevSettings) => ({
      ...prevSettings,
      ...importedSettings,
    }))
  }

  return (
    <SettingsContext.Provider value={{ settings, updateSetting, resetSettings, exportSettings, importSettings }}>
      {children}
    </SettingsContext.Provider>
  )
}

export const useSettings = () => useContext(SettingsContext)
