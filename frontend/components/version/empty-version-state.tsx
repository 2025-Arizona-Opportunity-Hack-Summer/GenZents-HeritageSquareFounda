import { memo } from "react"
import { History } from "lucide-react"

export const EmptyVersionState = memo(function EmptyVersionState() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center space-y-4">
        <History className="h-12 w-12 mx-auto text-muted-foreground" />
        <h3 className="text-lg font-semibold">No Version Selected</h3>
        <p className="text-muted-foreground">Select a version from the list to view details</p>
      </div>
    </div>
  )
})
