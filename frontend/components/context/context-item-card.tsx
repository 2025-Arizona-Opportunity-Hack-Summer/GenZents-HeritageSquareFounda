import { memo } from "react"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ContextItem } from "../../lib/types/context"
import { getTypeIcon, getTypeColor } from "../../lib/utils/context"

interface ContextItemCardProps {
  item: ContextItem
  isSelected: boolean
  onToggleSelection: (id: string) => void
  onRemove: (id: string) => void
}

export const ContextItemCard = memo<ContextItemCardProps>(function ContextItemCard({
  item,
  isSelected,
  onToggleSelection,
  onRemove,
}) {
  const Icon = getTypeIcon(item.type)

  return (
    <Card
      className={`cursor-pointer transition-colors ${isSelected ? "border-primary bg-primary/5" : ""}`}
      onClick={() => onToggleSelection(item.id)}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Icon className="h-4 w-4 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <p className="font-medium truncate">{item.name}</p>
              <p className="text-sm text-muted-foreground truncate">{item.path}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Badge className={`text-xs ${getTypeColor(item.type)}`}>{item.type}</Badge>

            {item.autoAttach && (
              <Badge variant="outline" className="text-xs">
                Auto
              </Badge>
            )}

            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                onRemove(item.id)
              }}
              className="h-6 w-6 p-0"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {item.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {item.tags.map((tag) => (
              <Badge key={tag} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
})
