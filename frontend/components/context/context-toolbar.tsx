import { memo } from "react"
import { Search, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ContextItemType } from "../../lib/types/context"
import { CONTEXT_ITEM_TYPES, FILTER_OPTIONS } from "./constants"

interface ContextToolbarProps {
  searchQuery: string
  filterType: string
  newItemPath: string
  newItemType: ContextItemType
  onSearchChange: (query: string) => void
  onFilterChange: (type: string) => void
  onNewItemPathChange: (path: string) => void
  onNewItemTypeChange: (type: ContextItemType) => void
  onAddContext: () => void
}

export const ContextToolbar = memo<ContextToolbarProps>(function ContextToolbar({
  searchQuery,
  filterType,
  newItemPath,
  newItemType,
  onSearchChange,
  onFilterChange,
  onNewItemPathChange,
  onNewItemTypeChange,
  onAddContext,
}) {
  return (
    <div className="space-y-4">
      {/* Search and Filter */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search context items..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-8"
          />
        </div>
        <Select value={filterType} onValueChange={onFilterChange}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {FILTER_OPTIONS.map(({ value, label }) => (
              <SelectItem key={value} value={value}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Add New Context */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Add Context Item</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <Select value={newItemType} onValueChange={onNewItemTypeChange}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CONTEXT_ITEM_TYPES.map(({ value, label }) => (
                  <SelectItem key={value} value={value}>
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input
              placeholder="Enter path or reference..."
              value={newItemPath}
              onChange={(e) => onNewItemPathChange(e.target.value)}
              className="flex-1"
            />
            <Button onClick={onAddContext} size="sm">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
})
