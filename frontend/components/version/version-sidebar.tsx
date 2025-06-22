import { memo } from "react"
import { History } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Version } from "../../lib/types/version"
import { VersionCard } from "./version-card"

interface VersionSidebarProps {
  searchQuery: string
  filterBranch: string
  branches: string[]
  filteredVersions: Version[]
  selectedVersion: Version | null
  onSearchChange: (query: string) => void
  onFilterChange: (branch: string) => void
  onVersionSelect: (version: Version) => void
}

export const VersionSidebar = memo<VersionSidebarProps>(function VersionSidebar({
  searchQuery,
  filterBranch,
  branches,
  filteredVersions,
  selectedVersion,
  onSearchChange,
  onFilterChange,
  onVersionSelect,
}) {
  return (
    <div className="w-80 border-r bg-muted/30">
      <div className="p-4 border-b">
        <h3 className="font-semibold flex items-center gap-2">
          <History className="h-5 w-5" />
          Version History
        </h3>
        <p className="text-sm text-muted-foreground">Track and manage version changes</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Filters */}
        <div className="space-y-2">
          <Input
            placeholder="Search versions..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
          />
          <Select value={filterBranch} onValueChange={onFilterChange}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Branches</SelectItem>
              {branches.map((branch) => (
                <SelectItem key={branch} value={branch}>
                  {branch}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Version List */}
        <ScrollArea className="h-[calc(100vh-300px)]">
          <div className="space-y-2">
            {filteredVersions.map((version) => (
              <VersionCard
                key={version.id}
                version={version}
                isSelected={selectedVersion?.id === version.id}
                onSelect={onVersionSelect}
              />
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
})
