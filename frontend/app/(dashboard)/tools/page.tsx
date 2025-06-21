import { Metadata } from "next"
import { Code, Folder, Sparkles } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScriptRunner } from "@/components/tools/script-runner"
import { FileOrganizer } from "@/components/tools/file-organizer"

export const metadata: Metadata = {
  title: "AI Tools",
  description: "Powerful tools to enhance your productivity",
  keywords: ["tools", "productivity", "AI"],
  openGraph: {
    title: "AI Tools",
    description: "Powerful tools to enhance your productivity",
    url: "/tools",
    type: "website",
  },
}

export default function ToolsPage() {
  return (
    <div className="h-full overflow-auto">
      <div className="container flex flex-col mx-auto px-6 py-4">

        <Tabs defaultValue="script-runner" className="flex-1">
          <TabsList className="grid w-full grid-cols-2 m-4">
            <TabsTrigger value="script-runner" className="flex items-center gap-2">
              <Code className="h-4 w-4" />
              Script Runner
            </TabsTrigger>
            <TabsTrigger value="file-organizer" className="flex items-center gap-2">
              <Folder className="h-4 w-4" />
              File Organizer
            </TabsTrigger>
          </TabsList>

          <TabsContent value="script-runner" className="flex-1 mt-0">
            <ScriptRunner />
          </TabsContent>

          <TabsContent value="file-organizer" className="flex-1 mt-0">
            <FileOrganizer />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
