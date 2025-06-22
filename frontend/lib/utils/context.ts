import { Hash, File, Folder, Code, Link, Database } from "lucide-react"
import { ContextItemType } from "../types/context"

// Memoized utility functions for better performance
export const getTypeIcon = (type: ContextItemType) => {
  switch (type) {
    case "file":
      return File
    case "folder":
      return Folder
    case "symbol":
      return Code
    case "selection":
      return Hash
    case "url":
      return Link
    case "database":
      return Database
    default:
      return File
  }
}

export const getTypeColor = (type: ContextItemType) => {
  switch (type) {
    case "file":
      return "bg-blue-100 text-blue-800"
    case "folder":
      return "bg-green-100 text-green-800"
    case "symbol":
      return "bg-purple-100 text-purple-800"
    case "selection":
      return "bg-orange-100 text-orange-800"
    case "url":
      return "bg-cyan-100 text-cyan-800"
    case "database":
      return "bg-red-100 text-red-800"
    default:
      return "bg-gray-100 text-gray-800"
  }
}

export const createNewContextItem = (path: string, type: ContextItemType) => ({
  id: Date.now().toString(),
  type,
  name: path.split("/").pop() || path,
  path,
  tags: [],
  lastUsed: new Date(),
})

export const filterContextItems = (items: any[], searchQuery: string, filterType: string) => {
  return items.filter((item) => {
    const matchesSearch =
      item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.tags.some((tag: string) => tag.toLowerCase().includes(searchQuery.toLowerCase()))

    const matchesType = filterType === "all" || item.type === filterType

    return matchesSearch && matchesType
  })
}
