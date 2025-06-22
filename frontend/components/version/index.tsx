"use client"

import { useVersionHistory } from "../../hooks/use-version-history"
import { VersionSidebar } from "./version-sidebar"
import { VersionDetails } from "./version-details"
import { EmptyVersionState } from "./empty-version-state"

export function VersionHistory() {
  const {
    versions,
    selectedVersion,
    compareVersion,
    filterBranch,
    searchQuery,
    branches,
    filteredVersions,
    setSelectedVersion,
    setCompareVersion,
    setFilterBranch,
    setSearchQuery,
    handleRollback,
  } = useVersionHistory()

  return (
    <div className="flex h-full">
      <VersionSidebar
        searchQuery={searchQuery}
        filterBranch={filterBranch}
        branches={branches}
        filteredVersions={filteredVersions}
        selectedVersion={selectedVersion}
        onSearchChange={setSearchQuery}
        onFilterChange={setFilterBranch}
        onVersionSelect={setSelectedVersion}
      />

      <div className="flex-1 flex flex-col">
        {selectedVersion ? (
          <VersionDetails
            selectedVersion={selectedVersion}
            compareVersion={compareVersion}
            versions={versions}
            onRollback={handleRollback}
            onCompareVersionChange={setCompareVersion}
          />
        ) : (
          <EmptyVersionState />
        )}
      </div>
    </div>
  )
}
