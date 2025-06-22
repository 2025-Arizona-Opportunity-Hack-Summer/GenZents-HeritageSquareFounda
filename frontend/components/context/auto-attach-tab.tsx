import * as React from "react"
import { memo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { ContextItem } from "../../lib/types/context"
import { getTypeIcon } from "../../lib/utils/context"

interface AutoAttachTabProps {
  contextItems: ContextItem[]
  onToggleAutoAttach: (id: string) => void
}

export const AutoAttachTab = memo<AutoAttachTabProps>(function AutoAttachTab({ contextItems, onToggleAutoAttach }) {
  const autoAttachItems = contextItems.filter((item) => item.autoAttach)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Auto-Attachment Rules</CardTitle>
        <CardDescription>Configure automatic context attachment based on conditions</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {autoAttachItems.map((item) => (
          <div key={item.id} className="flex items-center justify-between p-3 border rounded">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                {React.createElement(getTypeIcon(item.type), { className: "h-4 w-4" })}
                <span className="font-medium">{item.name}</span>
              </div>
              <Badge variant="outline" className="text-xs">
                {item.type}
              </Badge>
            </div>

            <Switch checked={item.autoAttach} onCheckedChange={() => onToggleAutoAttach(item.id)} />
          </div>
        ))}

        {autoAttachItems.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-4">No auto-attachment rules configured</p>
        )}
      </CardContent>
    </Card>
  )
})
