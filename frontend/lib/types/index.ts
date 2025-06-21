// Export all types from centralized type files
export * from "./ai"
export * from "./chat"
export * from "./context"
export * from "./database"
export * from "./features"
export * from "./ui"
export * from "./user"
export * from "./utils"
export * from "./version"
export * from "./workspace"

// Common utility types
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>

// Generic response types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface PaginatedResponse<T = any> {
  data: T[]
  total: number
  page: number
  limit: number
  hasNext: boolean
  hasPrevious: boolean
}

// Common UI component prop types
export interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
}

// Status types used across the application
export type LoadingState = "idle" | "loading" | "success" | "error"
export type SortOrder = "asc" | "desc"
export type ViewMode = "list" | "grid" | "table"
