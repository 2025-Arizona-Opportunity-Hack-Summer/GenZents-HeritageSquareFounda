import { memo } from "react"
import { RotateCcw, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Version } from "@/lib/types/version"
import { VersionDetailsTab } from "./version-details-tab"
import { VersionChangesTab } from "./version-changes-tab"
import { VersionCompareTab } from "./version-compare-tab"
import { exportVersion } from "@/lib/utils/version"

interface VersionDetailsProps {
  selectedVersion: Version
  compareVersion: Version | null
  versions: Version[]
  onRollback: (version: Version) => void
  onCompareVersionChange: (version: Version | null) => void
}

export const VersionDetails = memo<VersionDetailsProps>(function VersionDetails({
  selectedVersion,
  compareVersion,
  versions,
  onRollback,
  onCompareVersionChange,
}) {
  return (
    <Tabs defaultValue="details" className="flex-1">
      <div className="border-b px-4 py-2">
        <div className="flex items-center justify-between">
          <TabsList>
            <TabsTrigger value="details">Details</TabsTrigger>
            <TabsTrigger value="changes">Changes</TabsTrigger>
            <TabsTrigger value="compare">Compare</TabsTrigger>
          </TabsList>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => exportVersion(selectedVersion)}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            {selectedVersion.status !== "current" && (
              <Button variant="default" size="sm" onClick={() => onRollback(selectedVersion)}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Rollback
              </Button>
            )}
          </div>
        </div>
      </div>

      <TabsContent value="details" className="flex-1 p-4">
        <VersionDetailsTab version={selectedVersion} />
      </TabsContent>

      <TabsContent value="changes" className="flex-1 p-4">
        <VersionChangesTab changes={selectedVersion.changes} />
      </TabsContent>

      <TabsContent value="compare" className="flex-1 p-4">
        <VersionCompareTab
          selectedVersion={selectedVersion}
          compareVersion={compareVersion}
          versions={versions}
          onCompareVersionChange={onCompareVersionChange}
        />
      </TabsContent>
    </Tabs>
  )
})
