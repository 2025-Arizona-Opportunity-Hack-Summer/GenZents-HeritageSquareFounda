import { memo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"

export const WorkspaceTab = memo(function WorkspaceTab() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Workspace Configuration</CardTitle>
        <CardDescription>Configure workspace-wide context settings</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Workspace Root</Label>
          <Input placeholder="/path/to/workspace" />
        </div>{" "}
        <div className="space-y-2">
          <Label>Ignore Patterns</Label>
          <Textarea
            placeholder="node_modules/
*.log
.git/"
            rows={3}
          />
        </div>
        <div className="space-y-2">
          <Label>Include Extensions</Label>
          <Input placeholder=".ts,.tsx,.js,.jsx,.py,.md" />
        </div>
        <div className="flex items-center justify-between">
          <div>
            <Label>Auto-detect Context</Label>
            <p className="text-sm text-muted-foreground">Automatically detect relevant files</p>
          </div>
          <Switch />
        </div>
        <div className="flex items-center justify-between">
          <div>
            <Label>Include Git History</Label>
            <p className="text-sm text-muted-foreground">Include recent commits in context</p>
          </div>
          <Switch />
        </div>
      </CardContent>
    </Card>
  )
})
