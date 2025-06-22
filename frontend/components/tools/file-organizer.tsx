"use client"

import * as React from "react"
import Image from "next/image"
import {
  Folder,
  File,
  ImageIcon,
  FileText,
  Archive,
  Music,
  Video,
  Code,
  Settings,
  Search,
  Grid,
  List,
  Upload,
  Download,
  Trash2,
  Copy,
  Move,
  Eye,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Checkbox } from "@/components/ui/checkbox"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { logger } from "@/lib/utils/logger"
import type { FileItem } from "@/lib/types/user"

// Remove duplicate FileItem interface as it now comes from centralized types

const sampleFiles: FileItem[] = [
  {
    id: "1",
    name: "project-docs",
    type: "folder",
    path: "/documents/project-docs",
    lastModified: new Date("2024-01-15"),
    category: "documents",
  },
  {
    id: "2",
    name: "report.pdf",
    type: "file",
    size: 2048576,
    extension: "pdf",
    path: "/documents/report.pdf",
    lastModified: new Date("2024-01-14"),
    category: "documents",
    tags: ["report", "business"],
  },
  {
    id: "3",
    name: "image.jpg",
    type: "file",
    size: 1024000,
    extension: "jpg",
    path: "/images/image.jpg",
    lastModified: new Date("2024-01-13"),
    category: "images",
    preview: "/placeholder.svg?height=100&width=100",
  },
  {
    id: "4",
    name: "script.js",
    type: "file",
    size: 4096,
    extension: "js",
    path: "/code/script.js",
    lastModified: new Date("2024-01-12"),
    category: "code",
    tags: ["javascript", "frontend"],
  },
]

const categories = [
  { id: "all", name: "All Files", icon: File },
  { id: "documents", name: "Documents", icon: FileText },
  { id: "images", name: "Images", icon: ImageIcon },
  { id: "code", name: "Code", icon: Code },
  { id: "archives", name: "Archives", icon: Archive },
  { id: "media", name: "Media", icon: Music },
]

export function FileOrganizer() {
  const [files, setFiles] = React.useState<FileItem[]>(sampleFiles)
  const [selectedFiles, setSelectedFiles] = React.useState<string[]>([])
  const [viewMode, setViewMode] = React.useState<"grid" | "list">("grid")
  const [selectedCategory, setSelectedCategory] = React.useState("all")
  const [searchQuery, setSearchQuery] = React.useState("")
  const [sortBy, setSortBy] = React.useState("name")
  const [isProcessing, setIsProcessing] = React.useState(false)
  const [progress, setProgress] = React.useState(0)

  const filteredFiles = React.useMemo(() => {
    return files.filter((file) => {
      const matchesCategory = selectedCategory === "all" || file.category === selectedCategory
      const matchesSearch =
        file.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        file.tags?.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      return matchesCategory && matchesSearch
    })
  }, [files, selectedCategory, searchQuery])

  const sortedFiles = React.useMemo(() => {
    return [...filteredFiles].sort((a, b) => {
      switch (sortBy) {
        case "name":
          return a.name.localeCompare(b.name)
        case "size":
          return (b.size || 0) - (a.size || 0)
        case "date":
          return b.lastModified.getTime() - a.lastModified.getTime()
        default:
          return 0
      }
    })
  }, [filteredFiles, sortBy])

  const getFileIcon = (file: FileItem) => {
    if (file.type === "folder") return Folder

    switch (file.extension) {
      case "jpg":
      case "jpeg":
      case "png":
      case "gif":
        return ImageIcon
      case "pdf":
      case "doc":
      case "docx":
        return FileText
      case "js":
      case "ts":
      case "jsx":
      case "tsx":
        return Code
      case "zip":
      case "rar":
        return Archive
      case "mp3":
      case "wav":
        return Music
      case "mp4":
      case "avi":
        return Video
      default:
        return File
    }
  }

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return ""
    const sizes = ["B", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`
  }

  const handleSelectFile = (fileId: string) => {
    setSelectedFiles((prev) => (prev.includes(fileId) ? prev.filter((id) => id !== fileId) : [...prev, fileId]))
  }

  const handleSelectAll = () => {
    setSelectedFiles(selectedFiles.length === sortedFiles.length ? [] : sortedFiles.map((file) => file.id))
  }

  const handleOrganizeFiles = async () => {
    setIsProcessing(true)
    setProgress(0)

    // Simulate file organization
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsProcessing(false)
          return 100
        }
        return prev + 10
      })
    }, 200)
  }

  const handleBulkAction = (action: string) => {
    logger.info(`Performing ${action} on files`, { component: "file-organizer", selectedFiles })
    // Implement bulk actions here
  }

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="w-64 border-r bg-muted/30">
        <div className="p-4 border-b">
          <h3 className="font-semibold">File Organizer</h3>
          <p className="text-sm text-muted-foreground">Organize and manage your files</p>
        </div>

        <div className="p-4 space-y-4">
          {/* Categories */}
          <div>
            <Label className="text-sm font-medium">Categories</Label>
            <div className="mt-2 space-y-1">
              {categories.map((category) => {
                const Icon = category.icon
                const count =
                  category.id === "all" ? files.length : files.filter((f) => f.category === category.id).length

                return (
                  <Button
                    key={category.id}
                    variant={selectedCategory === category.id ? "secondary" : "ghost"}
                    className="w-full justify-start"
                    onClick={() => setSelectedCategory(category.id)}
                  >
                    <Icon className="h-4 w-4 mr-2" />
                    {category.name}
                    <Badge variant="outline" className="ml-auto">
                      {count}
                    </Badge>
                  </Button>
                )
              })}
            </div>
          </div>

          <Separator />

          {/* Organization Tools */}
          <div>
            <Label className="text-sm font-medium">Quick Actions</Label>
            <div className="mt-2 space-y-2">
              <Button
                variant="outline"
                size="sm"
                className="w-full justify-start"
                onClick={handleOrganizeFiles}
                disabled={isProcessing}
              >
                <Settings className="h-4 w-4 mr-2" />
                Auto Organize
              </Button>
              <Button variant="outline" size="sm" className="w-full justify-start">
                <Upload className="h-4 w-4 mr-2" />
                Upload Files
              </Button>
              <Button variant="outline" size="sm" className="w-full justify-start">
                <Download className="h-4 w-4 mr-2" />
                Export List
              </Button>
            </div>
          </div>

          {isProcessing && (
            <div className="space-y-2">
              <Label className="text-sm font-medium">Processing...</Label>
              <Progress value={progress} />
              <p className="text-xs text-muted-foreground">{progress}% complete</p>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="border-b p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search files..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8 w-64"
                />
              </div>

              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="size">Size</SelectItem>
                  <SelectItem value="date">Date</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-2">
              {selectedFiles.length > 0 && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      Actions ({selectedFiles.length})
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem onClick={() => handleBulkAction("copy")}>
                      <Copy className="h-4 w-4 mr-2" />
                      Copy
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleBulkAction("move")}>
                      <Move className="h-4 w-4 mr-2" />
                      Move
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleBulkAction("delete")}>
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}

              <Button variant="outline" size="sm" onClick={() => setViewMode(viewMode === "grid" ? "list" : "grid")}>
                {viewMode === "grid" ? <List className="h-4 w-4" /> : <Grid className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        </div>

        {/* File List */}
        <ScrollArea className="flex-1">
          <div className="p-4">
            {/* Select All */}
            <div className="flex items-center gap-2 mb-4">
              <Checkbox
                checked={selectedFiles.length === sortedFiles.length && sortedFiles.length > 0}
                onCheckedChange={handleSelectAll}
              />
              <Label className="text-sm">Select All ({sortedFiles.length} files)</Label>
            </div>

            {/* Files */}
            {viewMode === "grid" ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                {sortedFiles.map((file) => {
                  const Icon = getFileIcon(file)
                  return (
                    <Card
                      key={file.id}
                      className={`cursor-pointer transition-colors ${
                        selectedFiles.includes(file.id) ? "border-primary bg-primary/5" : ""
                      }`}
                      onClick={() => handleSelectFile(file.id)}
                    >
                      <CardContent className="p-4 text-center">
                        <div className="flex items-center justify-between mb-2">
                          <Checkbox checked={selectedFiles.includes(file.id)} onClick={(e) => e.stopPropagation()} />
                          <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                            <Eye className="h-3 w-3" />
                          </Button>
                        </div>{" "}
                        {file.preview ? (
                          <Image
                            src={file.preview || "/placeholder.svg"}
                            alt={file.name}
                            width={64}
                            height={64}
                            className="w-16 h-16 mx-auto mb-2 rounded object-cover"
                          />
                        ) : (
                          <Icon className="w-16 h-16 mx-auto mb-2 text-muted-foreground" />
                        )}
                        <p className="text-sm font-medium truncate">{file.name}</p>
                        {file.size && <p className="text-xs text-muted-foreground">{formatFileSize(file.size)}</p>}
                        {file.tags && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {file.tags.slice(0, 2).map((tag) => (
                              <Badge key={tag} variant="secondary" className="text-xs">
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            ) : (
              <div className="space-y-2">
                {sortedFiles.map((file) => {
                  const Icon = getFileIcon(file)
                  return (
                    <Card
                      key={file.id}
                      className={`cursor-pointer transition-colors ${
                        selectedFiles.includes(file.id) ? "border-primary bg-primary/5" : ""
                      }`}
                      onClick={() => handleSelectFile(file.id)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-center gap-4">
                          <Checkbox checked={selectedFiles.includes(file.id)} onClick={(e) => e.stopPropagation()} />{" "}
                          {file.preview ? (
                            <Image
                              src={file.preview || "/placeholder.svg"}
                              alt={file.name}
                              width={40}
                              height={40}
                              className="w-10 h-10 rounded object-cover"
                            />
                          ) : (
                            <Icon className="w-10 h-10 text-muted-foreground" />
                          )}
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{file.name}</p>
                            <p className="text-sm text-muted-foreground">{file.path}</p>
                          </div>
                          <div className="text-right">
                            {file.size && <p className="text-sm">{formatFileSize(file.size)}</p>}
                            <p className="text-xs text-muted-foreground">{file.lastModified.toLocaleDateString()}</p>
                          </div>
                          <Button variant="ghost" size="sm">
                            <Eye className="h-4 w-4" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            )}

            {sortedFiles.length === 0 && (
              <div className="text-center py-12">
                <File className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold">No files found</h3>
                <p className="text-muted-foreground">
                  {searchQuery ? `No files match "${searchQuery}"` : "No files in this category"}
                </p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
