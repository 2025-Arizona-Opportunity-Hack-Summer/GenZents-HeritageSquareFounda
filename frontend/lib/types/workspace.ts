import { ChatSettings } from "./chat"

export interface Workspace {
  id: string
  name: string
  description: string
  owner_id: string
  settings: WorkspaceSettings
  created_at: string
  updated_at: string
}

export interface WorkspaceSettings {
  visibility: "private" | "public"
  defaultChatSettings: ChatSettings
  branding: BrandingSettings
}

export interface BrandingSettings {
  logo?: string
  primaryColor: string
  secondaryColor: string
  customCSS?: string
}

export interface Analytics {
  user_id: string
  workspace_id?: string
  event: string
  properties: Record<string, any>
  timestamp: string
  session_id: string
  ip_address?: string
  user_agent?: string
}

export interface APIKey {
  id: string
  name: string
  key: string
  user_id: string
  scopes: string[]
  rate_limit: number
  expires_at?: string
  last_used?: string
  created_at: string
}

export interface Integration {
  id: string
  name: string
  type: string
  config: IntegrationConfig
  enabled: boolean
  user_id: string
  workspace_id?: string
}

export interface IntegrationConfig {
  [key: string]: any
}
