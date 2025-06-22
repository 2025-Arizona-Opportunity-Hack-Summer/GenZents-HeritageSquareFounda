import { memo } from "react"
import { GitBranch } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Version } from "@/lib/types/version"
import { getStatusColor } from "@/lib/utils/version"

interface VersionDetailsTabProps {
  version: Version
}

export const VersionDetailsTab = memo<VersionDetailsTabProps>(function VersionDetailsTab({ version }) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              Version {version.version}
              <Badge className={getStatusColor(version.status)}>{version.status}</Badge>
            </CardTitle>
            {version.tag && <Badge variant="secondary">{version.tag}</Badge>}
          </div>
          <CardDescription>{version.title}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm">{version.description}</p>

          <Separator />

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <Label className="text-muted-foreground">Author</Label>
              <p className="font-medium">{version.author}</p>
            </div>
            <div>
              <Label className="text-muted-foreground">Date</Label>
              <p className="font-medium">{version.date.toLocaleString()}</p>
            </div>
            <div>
              <Label className="text-muted-foreground">Branch</Label>
              <p className="font-medium flex items-center gap-1">
                <GitBranch className="h-3 w-3" />
                {version.branch}
              </p>
            </div>
            <div>
              <Label className="text-muted-foreground">Changes</Label>
              <p className="font-medium">{version.changes.length} files</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Change Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="space-y-1">
              <div className="text-2xl font-bold text-green-600">
                {version.changes.reduce((sum, change) => sum + change.linesAdded, 0)}
              </div>
              <div className="text-sm text-muted-foreground">Lines Added</div>
            </div>
            <div className="space-y-1">
              <div className="text-2xl font-bold text-red-600">
                {version.changes.reduce((sum, change) => sum + change.linesRemoved, 0)}
              </div>
              <div className="text-sm text-muted-foreground">Lines Removed</div>
            </div>
            <div className="space-y-1">
              <div className="text-2xl font-bold text-blue-600">{version.changes.length}</div>
              <div className="text-sm text-muted-foreground">Files Changed</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
})
