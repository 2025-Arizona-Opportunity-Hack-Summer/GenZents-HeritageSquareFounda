import { Metadata } from "next"
import { ContextManager } from "@/components/context"

export const metadata: Metadata = {
  title: "Context",
  description: "Manage your context settings.",
  keywords: ["context", "settings"],
  openGraph: {
    title: "Context",
    description: "Manage your context settings.",
    url: "/context",
    type: "website",
  },
}

export default function ContextPage() {
  return (
    <div className="h-full overflow-auto">
      <div className="container mx-auto px-4 py-8">
        <ContextManager />
      </div>
    </div>
  )
}
