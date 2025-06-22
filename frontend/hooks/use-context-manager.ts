import { useState, useMemo } from "react"
import { ContextItem, ContextItemType } from "../lib/types/context"
import { sampleContextItems } from "../components/context/constants"
import { createNewContextItem, filterContextItems } from "../lib/utils/context"

export const useContextManager = () => {
  const [contextItems, setContextItems] = useState<ContextItem[]>(sampleContextItems)
  const [selectedItems, setSelectedItems] = useState<string[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [filterType, setFilterType] = useState<string>("all")
  const [newItemPath, setNewItemPath] = useState("")
  const [newItemType, setNewItemType] = useState<ContextItemType>("file")

  const filteredItems = useMemo(() => {
    return filterContextItems(contextItems, searchQuery, filterType)
  }, [contextItems, searchQuery, filterType])

  const handleAddContext = () => {
    if (!newItemPath.trim()) return

    const newItem = createNewContextItem(newItemPath, newItemType)
    setContextItems((prev) => [newItem, ...prev])
    setNewItemPath("")
  }

  const handleRemoveContext = (id: string) => {
    setContextItems((prev) => prev.filter((item) => item.id !== id))
    setSelectedItems((prev) => prev.filter((itemId) => itemId !== id))
  }

  const handleToggleSelection = (id: string) => {
    setSelectedItems((prev) => (prev.includes(id) ? prev.filter((itemId) => itemId !== id) : [...prev, id]))
  }

  const handleToggleAutoAttach = (id: string) => {
    setContextItems((prev) => prev.map((item) => (item.id === id ? { ...item, autoAttach: !item.autoAttach } : item)))
  }

  return {
    contextItems,
    selectedItems,
    searchQuery,
    filterType,
    newItemPath,
    newItemType,
    filteredItems,
    setSearchQuery,
    setFilterType,
    setNewItemPath,
    setNewItemType,
    handleAddContext,
    handleRemoveContext,
    handleToggleSelection,
    handleToggleAutoAttach,
  }
}
