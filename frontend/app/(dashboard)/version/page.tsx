import { Metadata } from "next"
import { VersionHistory } from "@/components/version"

export const metadata: Metadata = {
  title: "Version",
  description: "View and manage your version history.",
  keywords: ["version", "history", "management"],
  openGraph: {
    title: "Version",
    description: "View and manage your version history.",
    url: "/version",
    type: "website",
  },
}

export default function VersionHistoryPage() {
  return (
    <div className="h-full overflow-auto">
      <div className="container mx-auto px-4 py-8">
        <VersionHistory />
      </div>
    </div>
  )
}
