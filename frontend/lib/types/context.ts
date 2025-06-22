export interface ContextItem {
  id: string
  type: "file" | "folder" | "symbol" | "selection" | "url" | "database"
  name: string
  path: string
  content?: string
  metadata?: Record<string, any>
  tags: string[]
  lastUsed: Date
  autoAttach?: boolean
}

export type ContextItemType = ContextItem["type"]

export interface ContextManagerState {
  contextItems: ContextItem[]
  selectedItems: string[]
  searchQuery: string
  filterType: string
  newItemPath: string
  newItemType: ContextItemType
}
