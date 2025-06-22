import { memo } from "react"
import { Eye } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Change } from "@/lib/types/version"
import { getChangeTypeColor } from "@/lib/utils/version"

interface ChangeItemProps {
  change: Change
}

const ChangeItem = memo<ChangeItemProps>(function ChangeItem({ change }) {
  return (
    <div className="flex items-center justify-between p-3 border rounded">
      <div className="flex items-center gap-3">
        <Badge className={getChangeTypeColor(change.type)}>{change.type}</Badge>
        <div>
          <p className="font-medium text-sm">{change.file}</p>
          <p className="text-xs text-muted-foreground">{change.description}</p>
        </div>
      </div>

      <div className="flex items-center gap-4 text-sm">
        {change.linesAdded > 0 && <span className="text-green-600">+{change.linesAdded}</span>}
        {change.linesRemoved > 0 && <span className="text-red-600">-{change.linesRemoved}</span>}
        <Button variant="ghost" size="sm">
          <Eye className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
})

interface VersionChangesTabProps {
  changes: Change[]
}

export const VersionChangesTab = memo<VersionChangesTabProps>(function VersionChangesTab({ changes }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>File Changes</CardTitle>
        <CardDescription>Detailed list of all changes in this version</CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96">
          <div className="space-y-3">
            {changes.map((change) => (
              <ChangeItem key={change.id} change={change} />
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
})
