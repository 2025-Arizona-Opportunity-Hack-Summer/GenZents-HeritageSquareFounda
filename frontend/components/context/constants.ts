import { ContextItem } from "../../lib/types/context"

export const sampleContextItems: ContextItem[] = [
  {
    id: "1",
    type: "file",
    name: "api.ts",
    path: "/src/lib/api.ts",
    content: "// API configuration and utilities",
    tags: ["api", "typescript"],
    lastUsed: new Date("2024-01-15"),
    autoAttach: true,
  },
  {
    id: "2",
    type: "folder",
    name: "components",
    path: "/src/components",
    tags: ["react", "ui"],
    lastUsed: new Date("2024-01-14"),
  },
  {
    id: "3",
    type: "symbol",
    name: "UserProvider",
    path: "/src/providers/user-provider.tsx:15",
    tags: ["react", "context"],
    lastUsed: new Date("2024-01-13"),
  },
]

export const CONTEXT_ITEM_TYPES = [
  { value: "file", label: "File" },
  { value: "folder", label: "Folder" },
  { value: "symbol", label: "Symbol" },
  { value: "selection", label: "Selection" },
  { value: "url", label: "URL" },
  { value: "database", label: "Database" },
] as const

export const FILTER_OPTIONS = [{ value: "all", label: "All Types" }, ...CONTEXT_ITEM_TYPES] as const
