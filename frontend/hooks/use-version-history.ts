import { useState, useMemo } from "react"
import { useToast } from "@/hooks/use-toast"
import { Version } from "../lib/types/version"
import { sampleVersions } from "../components/version/constants"
import { filterVersions, getBranches } from "@/lib/utils/version"

export const useVersionHistory = () => {
  const { toast } = useToast()
  const [versions, setVersions] = useState<Version[]>(sampleVersions)
  const [selectedVersion, setSelectedVersion] = useState<Version | null>(versions[0])
  const [compareVersion, setCompareVersion] = useState<Version | null>(null)
  const [filterBranch, setFilterBranch] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")

  const branches = useMemo(() => getBranches(versions), [versions])

  const filteredVersions = useMemo(() => {
    return filterVersions(versions, searchQuery, filterBranch)
  }, [versions, filterBranch, searchQuery])

  const handleRollback = async (version: Version) => {
    try {
      // Simulate rollback process
      toast({
        title: "Rolling back...",
        description: `Rolling back to version ${version.version}`,
      })

      // Update version status
      setVersions((prev) =>
        prev.map((v) => ({
          ...v,
          status: v.id === version.id ? "current" : v.status === "current" ? "previous" : v.status,
        }))
      )

      setTimeout(() => {
        toast({
          title: "Rollback completed",
          description: `Successfully rolled back to version ${version.version}`,
        })
      }, 2000)
    } catch (error) {
      toast({
        title: "Rollback failed",
        description: "Failed to rollback to the selected version",
        variant: "destructive",
      })
    }
  }

  return {
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
    toast,
  }
}
