import { memo } from "react"
import { GitBranch, Calendar, User } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Version } from "@/lib/types/version"
import { getStatusColor } from "@/lib/utils/version"

interface VersionCardProps {
  version: Version
  isSelected: boolean
  onSelect: (version: Version) => void
}

export const VersionCard = memo<VersionCardProps>(function VersionCard({ version, isSelected, onSelect }) {
  return (
    <Card
      className={`cursor-pointer transition-colors ${isSelected ? "border-primary bg-primary/5" : ""}`}
      onClick={() => onSelect(version)}
    >
      <CardContent className="p-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge variant="outline">{version.version}</Badge>
              <Badge className={getStatusColor(version.status)}>{version.status}</Badge>
            </div>
            {version.tag && (
              <Badge variant="secondary" className="text-xs">
                {version.tag}
              </Badge>
            )}
          </div>

          <h4 className="font-medium text-sm">{version.title}</h4>
          <p className="text-xs text-muted-foreground line-clamp-2">{version.description}</p>

          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <User className="h-3 w-3" />
            <span>{version.author}</span>
            <Calendar className="h-3 w-3 ml-2" />
            <span>{version.date.toLocaleDateString()}</span>
          </div>

          <div className="flex items-center gap-2">
            <GitBranch className="h-3 w-3 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">{version.branch}</span>
            <Badge variant="outline" className="text-xs ml-auto">
              {version.changes.length} changes
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  )
})
