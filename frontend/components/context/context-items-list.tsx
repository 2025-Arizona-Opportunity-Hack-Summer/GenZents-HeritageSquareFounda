import { memo } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ContextItem } from "../../lib/types/context"
import { ContextItemCard } from "./context-item-card"

interface ContextItemsListProps {
  items: ContextItem[]
  selectedItems: string[]
  onToggleSelection: (id: string) => void
  onRemoveItem: (id: string) => void
}

export const ContextItemsList = memo<ContextItemsListProps>(function ContextItemsList({
  items,
  selectedItems,
  onToggleSelection,
  onRemoveItem,
}) {
  return (
    <ScrollArea className="h-96">
      <div className="space-y-2">
        {items.map((item) => (
          <ContextItemCard
            key={item.id}
            item={item}
            isSelected={selectedItems.includes(item.id)}
            onToggleSelection={onToggleSelection}
            onRemove={onRemoveItem}
          />
        ))}
      </div>
    </ScrollArea>
  )
})
