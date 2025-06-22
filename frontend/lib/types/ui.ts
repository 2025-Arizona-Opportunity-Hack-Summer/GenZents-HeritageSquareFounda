import type React from "react"

// Theme-related types
export type Theme = "dark" | "light" | "system"

export interface ThemeProviderProps {
  children: React.ReactNode
  defaultTheme?: Theme
}

export interface ThemeProviderState {
  theme: Theme
  setTheme: (theme: Theme) => void
}

// Performance monitoring types
export interface PerformanceMetric {
  id: string
  name: string
  value: number
  unit: string
  status: "good" | "warning" | "critical"
  trend: "up" | "down" | "stable"
  history: { timestamp: Date; value: number }[]
}

export interface SystemHealth {
  overall: number
  cpu: number
  memory: number
  network: number
  storage: number
}

// Keyboard shortcuts types
export interface ShortcutCategory {
  id: string
  name: string
  shortcuts: KeyboardShortcut[]
}

export interface KeyboardShortcut {
  id: string
  name: string
  description: string
  keys: string[]
  action: string
  category: string
  customizable: boolean
}

// Auth component types
export interface AuthGuardProps {
  children: React.ReactNode
}

// Common component prop types
export interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
}

export interface MessageBubbleProps {
  message: import("./chat").Message
  isLast?: boolean
}

// Toast-related types
export interface ToastState {
  toasts: any[]
}

export interface ToastAction {
  type: string
  payload?: any
}
