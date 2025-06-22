export interface Change {
  id: string
  type: "added" | "modified" | "deleted"
  file: string
  description: string
  linesAdded: number
  linesRemoved: number
}

export interface Version {
  id: string
  version: string | number
  title: string
  description: string
  author?: string
  date: Date
  branch: string
  tag?: string
  changes: Change[]
  status: "current" | "previous" | "archived"
  // Additional fields from stores
  chatId?: string
  created_at?: string
  created_by?: string
  data?: any
  tags?: string[]
  parentVersion?: string
}

export interface VersionDiff {
  type: "added" | "modified" | "deleted"
  path: string
  oldValue?: any
  newValue?: any
}

export type ChangeType = Change["type"]
export type VersionStatus = Version["status"]

export interface VersionHistoryState {
  versions: Version[]
  selectedVersion: Version | null
  compareVersion: Version | null
  filterBranch: string
  searchQuery: string
}
