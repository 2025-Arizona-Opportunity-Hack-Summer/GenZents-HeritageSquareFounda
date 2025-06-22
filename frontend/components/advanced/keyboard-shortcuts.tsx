"use client"

import * as React from "react"
import { Keyboard, Search, Plus, Edit, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { useSettings } from "@/hooks/settings-provider"
import type { ShortcutCategory, KeyboardShortcut } from "@/lib/types/ui"

// Remove duplicate interfaces as they now come from centralized types

const defaultShortcuts: ShortcutCategory[] = [
  {
    id: "chat",
    name: "Chat",
    shortcuts: [
      {
        id: "new-chat",
        name: "New Chat",
        description: "Start a new conversation",
        keys: ["Ctrl", "N"],
        action: "newChat",
        category: "chat",
        customizable: true,
      },
      {
        id: "send-message",
        name: "Send Message",
        description: "Send the current message",
        keys: ["Enter"],
        action: "sendMessage",
        category: "chat",
        customizable: false,
      },
      {
        id: "new-line",
        name: "New Line",
        description: "Add a new line in message",
        keys: ["Shift", "Enter"],
        action: "newLine",
        category: "chat",
        customizable: false,
      },
      {
        id: "stop-generation",
        name: "Stop Generation",
        description: "Stop AI response generation",
        keys: ["Escape"],
        action: "stopGeneration",
        category: "chat",
        customizable: true,
      },
    ],
  },
  {
    id: "navigation",
    name: "Navigation",
    shortcuts: [
      {
        id: "toggle-sidebar",
        name: "Toggle Sidebar",
        description: "Show/hide the sidebar",
        keys: ["Ctrl", "B"],
        action: "toggleSidebar",
        category: "navigation",
        customizable: true,
      },
      {
        id: "search",
        name: "Search",
        description: "Open search dialog",
        keys: ["Ctrl", "K"],
        action: "search",
        category: "navigation",
        customizable: true,
      },
      {
        id: "settings",
        name: "Settings",
        description: "Open settings panel",
        keys: ["Ctrl", ","],
        action: "settings",
        category: "navigation",
        customizable: true,
      },
      {
        id: "help",
        name: "Help",
        description: "Show help dialog",
        keys: ["F1"],
        action: "help",
        category: "navigation",
        customizable: true,
      },
    ],
  },
  {
    id: "editing",
    name: "Editing",
    shortcuts: [
      {
        id: "copy-message",
        name: "Copy Message",
        description: "Copy selected message",
        keys: ["Ctrl", "C"],
        action: "copyMessage",
        category: "editing",
        customizable: false,
      },
      {
        id: "regenerate",
        name: "Regenerate Response",
        description: "Regenerate the last AI response",
        keys: ["Ctrl", "R"],
        action: "regenerate",
        category: "editing",
        customizable: true,
      },
      {
        id: "edit-message",
        name: "Edit Message",
        description: "Edit the selected message",
        keys: ["E"],
        action: "editMessage",
        category: "editing",
        customizable: true,
      },
      {
        id: "delete-message",
        name: "Delete Message",
        description: "Delete the selected message",
        keys: ["Delete"],
        action: "deleteMessage",
        category: "editing",
        customizable: true,
      },
    ],
  },
  {
    id: "tools",
    name: "Tools",
    shortcuts: [
      {
        id: "script-runner",
        name: "Script Runner",
        description: "Open script runner tool",
        keys: ["Ctrl", "Shift", "S"],
        action: "scriptRunner",
        category: "tools",
        customizable: true,
      },
      {
        id: "file-organizer",
        name: "File Organizer",
        description: "Open file organizer tool",
        keys: ["Ctrl", "Shift", "F"],
        action: "fileOrganizer",
        category: "tools",
        customizable: true,
      },
      {
        id: "prompt-enhancer",
        name: "Prompt Enhancer",
        description: "Open prompt enhancer tool",
        keys: ["Ctrl", "Shift", "P"],
        action: "promptEnhancer",
        category: "tools",
        customizable: true,
      },
    ],
  },
]

export function KeyboardShortcuts() {
  const { settings, updateSetting } = useSettings()
  const [shortcuts, setShortcuts] = React.useState<ShortcutCategory[]>(defaultShortcuts)
  const [searchQuery, setSearchQuery] = React.useState("")
  const [editingShortcut, setEditingShortcut] = React.useState<KeyboardShortcut | null>(null)
  const [recordingKeys, setRecordingKeys] = React.useState(false)
  const [recordedKeys, setRecordedKeys] = React.useState<string[]>([])

  const filteredShortcuts = React.useMemo(() => {
    if (!searchQuery) return shortcuts

    return shortcuts
      .map((category) => ({
        ...category,
        shortcuts: category.shortcuts.filter(
          (shortcut) =>
            shortcut.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            shortcut.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
            shortcut.keys.some((key) => key.toLowerCase().includes(searchQuery.toLowerCase()))
        ),
      }))
      .filter((category) => category.shortcuts.length > 0)
  }, [shortcuts, searchQuery])

  const handleEditShortcut = (shortcut: KeyboardShortcut) => {
    setEditingShortcut(shortcut)
    setRecordedKeys([...shortcut.keys])
  }

  const handleStartRecording = () => {
    setRecordingKeys(true)
    setRecordedKeys([])
  }

  const handleKeyDown = React.useCallback(
    (event: KeyboardEvent) => {
      if (!recordingKeys) return

      event.preventDefault()
      event.stopPropagation()

      const key = event.key
      const modifiers = []

      if (event.ctrlKey) modifiers.push("Ctrl")
      if (event.altKey) modifiers.push("Alt")
      if (event.shiftKey) modifiers.push("Shift")
      if (event.metaKey) modifiers.push("Cmd")

      const specialKeys = [
        "Enter",
        "Escape",
        "Space",
        "Tab",
        "Backspace",
        "Delete",
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
      ]
      const functionKeys = Array.from({ length: 12 }, (_, i) => `F${i + 1}`)

      if (specialKeys.includes(key) || functionKeys.includes(key)) {
        setRecordedKeys([...modifiers, key])
      } else if (key.length === 1 && modifiers.length > 0) {
        setRecordedKeys([...modifiers, key.toUpperCase()])
      }
    },
    [recordingKeys]
  )

  const handleStopRecording = () => {
    setRecordingKeys(false)
  }

  const handleSaveShortcut = () => {
    if (!editingShortcut || recordedKeys.length === 0) return

    setShortcuts((prev) =>
      prev.map((category) => ({
        ...category,
        shortcuts: category.shortcuts.map((shortcut) =>
          shortcut.id === editingShortcut.id ? { ...shortcut, keys: recordedKeys } : shortcut
        ),
      }))
    )

    // Update settings
    updateSetting("shortcuts", {
      ...settings.shortcuts,
      [editingShortcut.action]: recordedKeys.join("+"),
    })

    setEditingShortcut(null)
    setRecordedKeys([])
  }

  const handleResetShortcut = (shortcut: KeyboardShortcut) => {
    const defaultShortcut = defaultShortcuts.flatMap((cat) => cat.shortcuts).find((s) => s.id === shortcut.id)

    if (defaultShortcut) {
      setShortcuts((prev) =>
        prev.map((category) => ({
          ...category,
          shortcuts: category.shortcuts.map((s) => (s.id === shortcut.id ? { ...s, keys: defaultShortcut.keys } : s)),
        }))
      )
    }
  }

  React.useEffect(() => {
    if (recordingKeys) {
      document.addEventListener("keydown", handleKeyDown)
      return () => document.removeEventListener("keydown", handleKeyDown)
    }
  }, [recordingKeys, handleKeyDown])

  const formatKeys = (keys: string[]) => {
    return keys.join(" + ")
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Keyboard className="h-5 w-5" />
            Keyboard Shortcuts
          </h3>
          <p className="text-sm text-muted-foreground">Customize keyboard shortcuts for faster navigation</p>
        </div>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <Plus className="h-4 w-4 mr-2" />
              Add Custom
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add Custom Shortcut</DialogTitle>
              <DialogDescription>Create a new keyboard shortcut</DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Name</Label>
                <Input placeholder="Shortcut name..." />
              </div>
              <div className="space-y-2">
                <Label>Description</Label>
                <Input placeholder="What does this shortcut do?" />
              </div>
              <div className="space-y-2">
                <Label>Keys</Label>
                <div className="flex gap-2">
                  <Input placeholder="Press keys..." readOnly />
                  <Button variant="outline" size="sm">
                    Record
                  </Button>
                </div>
              </div>
              <Button className="w-full">Add Shortcut</Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search shortcuts..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-8"
        />
      </div>

      {/* Shortcuts List */}
      <ScrollArea className="h-96">
        <div className="space-y-6">
          {filteredShortcuts.map((category) => (
            <Card key={category.id}>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">{category.name}</CardTitle>
                <CardDescription>{category.shortcuts.length} shortcuts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {category.shortcuts.map((shortcut) => (
                    <div key={shortcut.id} className="flex items-center justify-between p-3 border rounded">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <p className="font-medium text-sm">{shortcut.name}</p>
                          {!shortcut.customizable && (
                            <Badge variant="secondary" className="text-xs">
                              System
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">{shortcut.description}</p>
                      </div>

                      <div className="flex items-center gap-3">
                        <div className="flex gap-1">
                          {shortcut.keys.map((key, index) => (
                            <React.Fragment key={index}>
                              <Badge variant="outline" className="font-mono text-xs">
                                {key}
                              </Badge>
                              {index < shortcut.keys.length - 1 && (
                                <span className="text-muted-foreground text-xs">+</span>
                              )}
                            </React.Fragment>
                          ))}
                        </div>

                        {shortcut.customizable && (
                          <div className="flex gap-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleEditShortcut(shortcut)}
                              className="h-8 w-8 p-0"
                            >
                              <Edit className="h-3 w-3" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleResetShortcut(shortcut)}
                              className="h-8 w-8 p-0"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>

      {/* Edit Shortcut Dialog */}
      <Dialog open={!!editingShortcut} onOpenChange={() => setEditingShortcut(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Shortcut</DialogTitle>
            <DialogDescription>
              {editingShortcut?.name} - {editingShortcut?.description}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Current Keys</Label>
              <div className="flex gap-1 p-2 border rounded bg-muted/50">
                {recordedKeys.length > 0 ? (
                  recordedKeys.map((key, index) => (
                    <React.Fragment key={index}>
                      <Badge variant="outline" className="font-mono">
                        {key}
                      </Badge>
                      {index < recordedKeys.length - 1 && <span className="text-muted-foreground">+</span>}
                    </React.Fragment>
                  ))
                ) : (
                  <span className="text-muted-foreground text-sm">No keys recorded</span>
                )}
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                variant={recordingKeys ? "destructive" : "default"}
                onClick={recordingKeys ? handleStopRecording : handleStartRecording}
                className="flex-1"
              >
                {recordingKeys ? "Stop Recording" : "Record New Keys"}
              </Button>

              {recordedKeys.length > 0 && (
                <Button onClick={handleSaveShortcut} variant="outline">
                  Save
                </Button>
              )}
            </div>

            {recordingKeys && (
              <div className="p-3 bg-blue-50 border border-blue-200 rounded">
                <p className="text-sm text-blue-800">
                  Press the key combination you want to use. Press Escape to cancel.
                </p>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
