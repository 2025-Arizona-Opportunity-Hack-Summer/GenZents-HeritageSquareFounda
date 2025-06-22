"use client"

import * as React from "react"
import { Play, CircleStopIcon as Stop, Save, Upload, Download, Code } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { logger } from "@/lib/utils/logger"
import type { ScriptConfig } from "@/lib/types/user"

// Remove duplicate ScriptConfig interface as it now comes from centralized types

const sampleScripts: ScriptConfig[] = [
  {
    id: "1",
    name: "Data Processor",
    description: "Process CSV data and generate reports",
    language: "python",
    code: `import pandas as pd
import json

def process_data(file_path, output_format="json"):
    df = pd.read_csv(file_path)
    result = df.describe().to_dict()
    
    if output_format == "json":
        return json.dumps(result, indent=2)
    return result

# Execute
result = process_data("data.csv")
print(result)`,
    parameters: [
      { name: "file_path", type: "file", value: "", description: "Input CSV file" },
      { name: "output_format", type: "select", value: "json", description: "Output format" },
    ],
    tags: ["data", "csv", "analysis"],
    status: "idle",
  },
  {
    id: "2",
    name: "API Tester",
    description: "Test REST API endpoints",
    language: "javascript",
    code: `async function testAPI(endpoint, method = "GET", data = null) {
  try {
    const options = {
      method,
      headers: { "Content-Type": "application/json" }
    };
    
    if (data) options.body = JSON.stringify(data);
    
    const response = await fetch(endpoint, options);
    const result = await response.json();
    
    logger.debug("Status", { component: 'script-runner', status: response.status });
    logger.debug("Response", { component: 'script-runner', result });
    return result;
  } catch (error) {
    logger.error("Error", error as Error, { component: 'script-runner' });
  }
}

// Execute
testAPI("https://jsonplaceholder.typicode.com/posts/1");`,
    parameters: [
      { name: "endpoint", type: "text", value: "", description: "API endpoint URL" },
      { name: "method", type: "select", value: "GET", description: "HTTP method" },
    ],
    tags: ["api", "testing", "http"],
    status: "idle",
  },
]

export function ScriptRunner() {
  const [scripts, setScripts] = React.useState<ScriptConfig[]>(sampleScripts)
  const [selectedScript, setSelectedScript] = React.useState<ScriptConfig | null>(scripts[0])
  const [isRunning, setIsRunning] = React.useState(false)
  const [output, setOutput] = React.useState("")
  const [progress, setProgress] = React.useState(0)

  const handleRunScript = async () => {
    if (!selectedScript) return

    setIsRunning(true)
    setProgress(0)
    setOutput("Starting execution...\n")

    // Simulate script execution
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsRunning(false)
          setOutput((prev) => prev + "\nExecution completed successfully!")
          return 100
        }
        return prev + 10
      })
    }, 200)

    // Simulate output
    setTimeout(() => {
      setOutput((prev) => prev + `\nExecuting ${selectedScript.name}...\n`)
    }, 500)

    setTimeout(() => {
      setOutput((prev) => prev + "Processing parameters...\n")
    }, 1000)

    setTimeout(() => {
      setOutput((prev) => prev + "Running script logic...\n")
    }, 1500)
  }

  const handleStopScript = () => {
    setIsRunning(false)
    setProgress(0)
    setOutput((prev) => prev + "\nExecution stopped by user.")
  }

  const handleSaveScript = () => {
    if (!selectedScript) return
    // Save script logic here
    logger.info("Saving script", { component: "script-runner", selectedScript })
  }

  const updateParameter = (paramName: string, value: string) => {
    if (!selectedScript) return

    const updatedScript = {
      ...selectedScript,
      parameters: selectedScript.parameters.map((param) => (param.name === paramName ? { ...param, value } : param)),
    }
    setSelectedScript(updatedScript)
  }

  return (
    <div className="flex h-full">
      {/* Script Library */}
      <div className="w-80 border-r bg-muted/30">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Script Library</h3>
          <p className="text-sm text-muted-foreground">Manage and run your scripts</p>
        </div>
        <ScrollArea className="h-[calc(100vh-200px)]">
          <div className="p-4 space-y-2">
            {scripts.map((script) => (
              <Card
                key={script.id}
                className={`cursor-pointer transition-colors ${
                  selectedScript?.id === script.id ? "border-primary" : ""
                }`}
                onClick={() => setSelectedScript(script)}
              >
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm">{script.name}</CardTitle>
                    <Badge variant="outline" className="text-xs">
                      {script.language}
                    </Badge>
                  </div>
                  <CardDescription className="text-xs">{script.description}</CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex flex-wrap gap-1">
                    {script.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {selectedScript ? (
          <Tabs defaultValue="config" className="flex-1">
            <div className="border-b px-4 py-2">
              <div className="flex items-center justify-between">
                <TabsList>
                  <TabsTrigger value="config">Configuration</TabsTrigger>
                  <TabsTrigger value="code">Code Editor</TabsTrigger>
                  <TabsTrigger value="output">Output</TabsTrigger>
                </TabsList>
                <div className="flex items-center gap-2">
                  <Button
                    onClick={isRunning ? handleStopScript : handleRunScript}
                    variant={isRunning ? "destructive" : "default"}
                    size="sm"
                  >
                    {isRunning ? <Stop className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                    {isRunning ? "Stop" : "Run"}
                  </Button>
                  <Button onClick={handleSaveScript} variant="outline" size="sm">
                    <Save className="h-4 w-4 mr-2" />
                    Save
                  </Button>
                </div>
              </div>
            </div>

            <TabsContent value="config" className="flex-1 p-4">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold">{selectedScript.name}</h3>
                  <p className="text-muted-foreground">{selectedScript.description}</p>
                </div>

                <Separator />

                <div className="space-y-4">
                  <h4 className="font-medium">Parameters</h4>
                  {selectedScript.parameters.map((param) => (
                    <div key={param.name} className="space-y-2">
                      <Label htmlFor={param.name}>{param.name}</Label>
                      <p className="text-sm text-muted-foreground">{param.description}</p>
                      {param.type === "select" ? (
                        <Select value={param.value} onValueChange={(value) => updateParameter(param.name, value)}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="json">JSON</SelectItem>
                            <SelectItem value="csv">CSV</SelectItem>
                            <SelectItem value="xml">XML</SelectItem>
                          </SelectContent>
                        </Select>
                      ) : param.type === "file" ? (
                        <div className="flex gap-2">
                          <Input
                            id={param.name}
                            value={param.value}
                            onChange={(e) => updateParameter(param.name, e.target.value)}
                            placeholder="Select file..."
                          />
                          <Button variant="outline" size="sm">
                            <Upload className="h-4 w-4" />
                          </Button>
                        </div>
                      ) : (
                        <Input
                          id={param.name}
                          value={param.value}
                          onChange={(e) => updateParameter(param.name, e.target.value)}
                          placeholder={`Enter ${param.name}...`}
                        />
                      )}
                    </div>
                  ))}
                </div>

                {isRunning && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Execution Progress</span>
                      <span className="text-sm text-muted-foreground">{progress}%</span>
                    </div>
                    <Progress value={progress} />
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="code" className="flex-1 p-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Code Editor</h4>
                  <Badge variant="outline">{selectedScript.language}</Badge>
                </div>
                <Textarea
                  value={selectedScript.code}
                  onChange={(e) => setSelectedScript({ ...selectedScript, code: e.target.value })}
                  className="font-mono text-sm min-h-[400px]"
                  placeholder="Enter your code here..."
                />
              </div>
            </TabsContent>

            <TabsContent value="output" className="flex-1 p-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Execution Output</h4>
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </Button>
                </div>
                <Card>
                  <CardContent className="p-4">
                    <pre className="text-sm font-mono whitespace-pre-wrap bg-muted p-4 rounded">
                      {output || "No output yet. Run a script to see results."}
                    </pre>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-4">
              <Code className="h-12 w-12 mx-auto text-muted-foreground" />
              <h3 className="text-lg font-semibold">No Script Selected</h3>
              <p className="text-muted-foreground">Select a script from the library to get started</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
