import { memo } from "react"
import { ContrastIcon as Compare } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Version } from "../../lib/types/version"

interface VersionCompareTabProps {
  selectedVersion: Version
  compareVersion: Version | null
  versions: Version[]
  onCompareVersionChange: (version: Version | null) => void
}

export const VersionCompareTab = memo<VersionCompareTabProps>(function VersionCompareTab({
  selectedVersion,
  compareVersion,
  versions,
  onCompareVersionChange,
}) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Compare Versions</CardTitle>
          <CardDescription>Select another version to compare changes</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Base Version</Label>
              <div className="p-2 border rounded bg-muted/50">
                <p className="font-medium">{selectedVersion.version}</p>
                <p className="text-sm text-muted-foreground">{selectedVersion.title}</p>
              </div>
            </div>
            <div>
              <Label>Compare With</Label>
              <Select
                value={compareVersion?.id || ""}
                onValueChange={(value) => {
                  const version = versions.find((v) => v.id === value)
                  onCompareVersionChange(version || null)
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select version..." />
                </SelectTrigger>
                <SelectContent>
                  {versions
                    .filter((v) => v.id !== selectedVersion.id)
                    .map((version) => (
                      <SelectItem key={version.id} value={version.id}>
                        {version.version} - {version.title}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {compareVersion && (
            <Button className="w-full">
              <Compare className="h-4 w-4 mr-2" />
              Compare Versions
            </Button>
          )}
        </CardContent>
      </Card>

      {compareVersion && (
        <Card>
          <CardHeader>
            <CardTitle>Comparison Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <h4 className="font-medium">
                    {selectedVersion.version} ({selectedVersion.date.toLocaleDateString()})
                  </h4>
                  <div className="space-y-1">
                    <p>Changes: {selectedVersion.changes.length}</p>
                    <p>Author: {selectedVersion.author}</p>
                  </div>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium">
                    {compareVersion.version} ({compareVersion.date.toLocaleDateString()})
                  </h4>
                  <div className="space-y-1">
                    <p>Changes: {compareVersion.changes.length}</p>
                    <p>Author: {compareVersion.author}</p>
                  </div>
                </div>
              </div>

              <Separator />

              <div className="text-center text-sm text-muted-foreground">
                Detailed diff comparison would be displayed here
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
})
