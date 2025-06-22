// Types for accessibility utilities
export interface KeyboardShortcutOptions {
  ctrl?: boolean
  alt?: boolean
  shift?: boolean
  meta?: boolean
}

// Types for rate limiting
export interface RateLimitOptions {
  windowMs: number
  max: number
  message: string
  standardHeaders: boolean
  legacyHeaders: boolean
}

export interface RateLimitStore {
  [key: string]: {
    count: number
    resetTime: number
  }
}

export interface RateLimitResult {
  allowed: boolean
  remaining: number
  resetTime: number
}

// Types for password validation
export interface PasswordValidationResult {
  valid: boolean
  errors: string[]
}

// Types for performance monitoring
export interface PerformanceMetrics {
  avg: number
  min: number
  max: number
  count: number
}

// Types for internationalization
export type TranslationDictionary = Record<string, Record<string, string>>

export interface I18nConfig {
  defaultLanguage: string
  supportedLanguages: string[]
  fallbackLanguage: string
}
