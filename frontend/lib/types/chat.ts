export interface Chat {
  id: string
  title: string
  description?: string
  created_at: string
  updated_at: string
  user_id: string
  settings: ChatSettings
  metadata: ChatMetadata
  messages: Message[]
  versions: ChatVersion[]
  tags: string[]
  category: string
  status: "active" | "archived" | "deleted"
  visibility: "private" | "shared" | "public"
  template: boolean
  attachments: Attachment[]
  contextFiles: ContextFile[]
  // Additional fields from stores
  model?: string
  system_prompt?: string
  bookmarked?: boolean
  archived?: boolean
  shared?: boolean
  version?: number
  parentVersion?: string
}

export interface ChatSettings {
  model: string
  temperature: number
  maxTokens: number
  systemPrompt: string
  contextWindow: number
  streaming: boolean
  autoSave: boolean
  encryption: boolean
  retentionDays: number
}

export interface ChatMetadata {
  totalMessages: number
  totalTokens: number
  averageResponseTime: number
  lastActivity: string
  language: string
  domain: string
}

export interface Message {
  id: string
  chat_id: string
  user_id?: string
  role: "user" | "assistant" | "system" | "function"
  content: string
  created_at: string
  updated_at?: string
  timestamp?: Date
  edited: boolean
  deleted: boolean
  metadata: MessageMetadata
  attachments: Attachment[]
  reactions: Reaction[]
  mentions: Mention[]
  thread_id?: string
  parent_id?: string
  parentId?: string
  status: MessageStatus
  encryption: EncryptionInfo
}

export interface MessageMetadata {
  model?: string
  tokens?: number
  cost?: number
  processingTime?: number
  confidence?: number
  sources?: Source[]
  citations?: Citation[]
  language?: string
  sentiment?: "positive" | "negative" | "neutral"
  topics?: string[]
  entities?: Entity[]
}

export interface MessageStatus {
  sent: boolean
  delivered: boolean
  read: boolean
  error?: string
  retries: number
  status?: "sending" | "sent" | "delivered" | "read" | "error"
}

export interface EncryptionInfo {
  encrypted: boolean
  algorithm?: string
  keyId?: string
}

export interface Attachment {
  id: string
  name: string
  type: string
  size: number
  url: string
  thumbnail?: string
  metadata: AttachmentMetadata
  virus_scan: VirusScanResult
  created_at: string
  uploadedAt?: string
  content?: string
  preview?: string
}

export interface AttachmentMetadata {
  width?: number
  height?: number
  duration?: number
  pages?: number
  encoding?: string
  checksum: string
}

export interface VirusScanResult {
  scanned: boolean
  clean: boolean
  threats?: string[]
  scanned_at: string
}

export interface Reaction {
  id: string
  user_id: string
  type: string
  emoji: string
  created_at: string
}

export interface Mention {
  id: string
  user_id: string
  start: number
  end: number
  type: "user" | "channel" | "file" | "url"
}

export interface Source {
  url: string
  title: string
  description: string
  domain: string
  confidence: number
}

export interface Citation {
  text: string
  source: string
  page?: number
  line?: number
}

export interface Entity {
  text: string
  type: string
  confidence: number
  start: number
  end: number
}

export interface ChatVersion {
  id: string
  version: number
  title: string
  description: string
  created_at: string
  created_by: string
  changes: VersionChange[]
  parent_version?: string
  branch: string
  tags: string[]
}

export interface VersionChange {
  type: "added" | "modified" | "deleted"
  resource: "message" | "setting" | "metadata"
  resource_id: string
  old_value?: any
  new_value?: any
}

export interface Comment {
  id: string
  messageId?: string
  message_id?: string
  userId?: string
  user_id?: string
  content: string
  created_at: string
  resolved: boolean
  replies: Comment[]
}

export interface ShareSettings {
  isPublic: boolean
  allowComments: boolean
  allowEditing: boolean
  expiresAt?: string
  password?: string
  domainRestriction?: string
}

export interface ChatSession {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
  tags?: string[]
  bookmarked?: boolean
  model?: string
  systemPrompt?: string
}

export interface ContextFile {
  id: string
  name: string
  type: string
  size: number
  path: string
  content?: string
  added_at: string
}
