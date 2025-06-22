import { memo } from "react"
import { ContextToolbar } from "./context-toolbar"
import { ContextItemsList } from "./context-items-list"
import { ContextItemType, ContextItem } from "../../lib/types/context"

interface ContextBrowserTabProps {
  searchQuery: string
  filterType: string
  newItemPath: string
  newItemType: ContextItemType
  filteredItems: ContextItem[]
  selectedItems: string[]
  onSearchChange: (query: string) => void
  onFilterChange: (type: string) => void
  onNewItemPathChange: (path: string) => void
  onNewItemTypeChange: (type: ContextItemType) => void
  onAddContext: () => void
  onToggleSelection: (id: string) => void
  onRemoveItem: (id: string) => void
}

export const ContextBrowserTab = memo<ContextBrowserTabProps>(function ContextBrowserTab({
  searchQuery,
  filterType,
  newItemPath,
  newItemType,
  filteredItems,
  selectedItems,
  onSearchChange,
  onFilterChange,
  onNewItemPathChange,
  onNewItemTypeChange,
  onAddContext,
  onToggleSelection,
  onRemoveItem,
}) {
  return (
    <div className="space-y-4">
      <ContextToolbar
        searchQuery={searchQuery}
        filterType={filterType}
        newItemPath={newItemPath}
        newItemType={newItemType}
        onSearchChange={onSearchChange}
        onFilterChange={onFilterChange}
        onNewItemPathChange={onNewItemPathChange}
        onNewItemTypeChange={onNewItemTypeChange}
        onAddContext={onAddContext}
      />

      <ContextItemsList
        items={filteredItems}
        selectedItems={selectedItems}
        onToggleSelection={onToggleSelection}
        onRemoveItem={onRemoveItem}
      />
    </div>
  )
})
