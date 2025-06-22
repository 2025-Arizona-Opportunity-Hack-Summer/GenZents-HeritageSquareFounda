"use client"

import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useContextManager } from "../../hooks/use-context-manager"
import { ContextBrowserTab } from "./context-browser-tab"
import { AutoAttachTab } from "./auto-attach-tab"
import { WorkspaceTab } from "./workspace-tab"

export function ContextManager() {
  const {
    searchQuery,
    filterType,
    newItemPath,
    newItemType,
    filteredItems,
    selectedItems,
    contextItems,
    setSearchQuery,
    setFilterType,
    setNewItemPath,
    setNewItemType,
    handleAddContext,
    handleToggleSelection,
    handleRemoveContext,
    handleToggleAutoAttach,
  } = useContextManager()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Context Manager</h3>
          <p className="text-sm text-muted-foreground">Manage files, folders, and symbols for AI context</p>
        </div>
        <Badge variant="secondary">{selectedItems.length} selected</Badge>
      </div>

      <Tabs defaultValue="browser" className="space-y-4">
        <TabsList>
          <TabsTrigger value="browser">Context Browser</TabsTrigger>
          <TabsTrigger value="auto-attach">Auto-Attach Rules</TabsTrigger>
          <TabsTrigger value="workspace">Workspace</TabsTrigger>
        </TabsList>

        <TabsContent value="browser" className="space-y-4">
          <ContextBrowserTab
            searchQuery={searchQuery}
            filterType={filterType}
            newItemPath={newItemPath}
            newItemType={newItemType}
            filteredItems={filteredItems}
            selectedItems={selectedItems}
            onSearchChange={setSearchQuery}
            onFilterChange={setFilterType}
            onNewItemPathChange={setNewItemPath}
            onNewItemTypeChange={setNewItemType}
            onAddContext={handleAddContext}
            onToggleSelection={handleToggleSelection}
            onRemoveItem={handleRemoveContext}
          />
        </TabsContent>

        <TabsContent value="auto-attach" className="space-y-4">
          <AutoAttachTab contextItems={contextItems} onToggleAutoAttach={handleToggleAutoAttach} />
        </TabsContent>

        <TabsContent value="workspace" className="space-y-4">
          <WorkspaceTab />
        </TabsContent>
      </Tabs>
    </div>
  )
}
