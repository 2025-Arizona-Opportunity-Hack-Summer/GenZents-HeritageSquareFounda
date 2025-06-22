"use client"

import { useState } from "react"
import {
  Settings,
  User,
  Palette,
  Shield,
  Download,
  Upload,
  RotateCcw,
  Link,
  UserCog,
  Trash2,
  LogOut,
  Cloud,
  PersonStanding,
} from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useSettings } from "@/hooks/settings-provider"
import { useTheme } from "@/hooks/theme-provider"
import { ThemeToggle } from "@/components/theme-toggle"
import { logger } from "@/lib/utils/logger"

const settingsSections = [
  { id: "general", label: "General", icon: Settings },
  { id: "appearance", label: "Appearance", icon: Palette },
  { id: "chat", label: "Chat", icon: User },
  { id: "connections", label: "Connections", icon: Link },
  { id: "privacy", label: "Privacy", icon: Shield },
]

export function SettingsDialog({ children }: { children: React.ReactNode }) {
  const [activeSection, setActiveSection] = useState("general")
  const { settings, updateSetting, resetSettings, exportSettings, importSettings } = useSettings()
  const { theme, setTheme } = useTheme()

  const handleFileImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const importedSettings = JSON.parse(e.target?.result as string)
          importSettings(importedSettings)
        } catch (error) {
          logger.error("Failed to import settings", error as Error, { component: "settings-dialog" })
        }
      }
      reader.readAsText(file)
    }
  }

  const renderSectionContent = () => {
    switch (activeSection) {
      case "general":
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium mb-4">General Settings</h3>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>Language</Label>
                  <Select value={settings.language} onValueChange={(value) => updateSetting("language", value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="es">Spanish</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center justify-between">
                  <Label>Show follow up suggestions in chats</Label>
                  <Switch
                    checked={settings.showFollowUpSuggestions}
                    onCheckedChange={(checked) => updateSetting("showFollowUpSuggestions", checked)}
                  />
                </div>

                <Separator />

                <div>
                  <h4 className="font-medium mb-4 text-destructive">Danger Zone</h4>
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Delete all chats</Label>
                        <p className="text-sm text-muted-foreground">
                          This will permanently delete all your chat history
                        </p>
                      </div>
                      <Button
                        variant="outline"
                        className="text-destructive"
                        onClick={() => {
                          if (confirm("Are you sure you want to delete all chats? This action cannot be undone.")) {
                            updateSetting("chats", [])
                          }
                        }}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete All Chats
                      </Button>
                    </div>

                    <div className="space-y-2">
                      <Label>Log out from all devices</Label>
                      <p className="text-sm text-muted-foreground">
                        Sign out from all devices where you&apos;re logged in
                      </p>
                      <Button
                        variant="destructive"
                        onClick={() => {
                          if (
                            confirm(
                              "Are you sure you want to log out from all devices? You&apos;ll need to sign in again on all your devices."
                            )
                          ) {
                            // Call your logout from all devices function here
                            logger.info("Logging out from all devices", { component: "settings-dialog" })
                            // This would typically invalidate all sessions
                          }
                        }}
                      >
                        <LogOut className="h-4 w-4 mr-2" />
                        Log Out From All Devices
                      </Button>
                    </div>

                    <div className="space-y-2">
                      <Label>Delete account</Label>
                      <p className="text-sm text-muted-foreground">
                        Permanently delete your account and all associated data
                      </p>
                      <Button
                        variant="destructive"
                        onClick={() => {
                          const confirmation = prompt(
                            "This action cannot be undone. Type 'DELETE' to confirm account deletion:"
                          )
                          if (confirmation === "DELETE") {
                            // Call your delete account function here
                            logger.info("Deleting account", { component: "settings-dialog" })
                            // This would typically delete all user data and sessions
                          } else if (confirmation !== null) {
                            alert("Account deletion cancelled. Please type 'DELETE' exactly to confirm.")
                          }
                        }}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete Account
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case "appearance":
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium mb-4">Appearance</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Theme</Label>
                  <ThemeToggle />
                </div>
                <div className="space-y-2">
                  <Label>Font Size</Label>
                  <Select value={settings.fontSize} onValueChange={(value: any) => updateSetting("fontSize", value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="small">Small</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="large">Large</SelectItem>
                    </SelectContent>{" "}
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Layout Density</Label>
                  <Select
                    value={settings.layoutDensity}
                    onValueChange={(value: any) => updateSetting("layoutDensity", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="compact">Compact</SelectItem>
                      <SelectItem value="comfortable">Comfortable</SelectItem>
                      <SelectItem value="spacious">Spacious</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>High Contrast</Label>
                    <p className="text-sm text-muted-foreground">Increase contrast for better visibility</p>
                  </div>
                  <Switch
                    checked={settings.highContrast}
                    onCheckedChange={(checked) => updateSetting("highContrast", checked)}
                  />
                </div>
              </div>
            </div>
          </div>
        )

      case "chat":
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium mb-4">Chat Settings</h3>
              <div className="space-y-6">
                {/* Personalization Section */}
                <div>
                  <h4 className="font-medium mb-4">Personalization</h4>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Custom Instructions</Label>
                      <p className="text-sm text-muted-foreground">
                        These instructions will be included in every conversation to customize the AI&apos;s responses
                      </p>
                      <Textarea
                        value={settings.customInstructions || ""}
                        onChange={(e) => updateSetting("customInstructions", e.target.value)}
                        placeholder="e.g., Always be concise, speak like a professional, focus on practical solutions..."
                        rows={4}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Communication Style</Label>
                      <Select
                        value={settings.communicationStyle || "balanced"}
                        onValueChange={(value) => updateSetting("communicationStyle", value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="professional">Professional</SelectItem>
                          <SelectItem value="casual">Casual</SelectItem>
                          <SelectItem value="friendly">Friendly</SelectItem>
                          <SelectItem value="balanced">Balanced</SelectItem>
                          <SelectItem value="technical">Technical</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Response Length Preference</Label>
                      <Select
                        value={settings.responseLength || "balanced"}
                        onValueChange={(value) => updateSetting("responseLength", value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="concise">Concise</SelectItem>
                          <SelectItem value="balanced">Balanced</SelectItem>
                          <SelectItem value="detailed">Detailed</SelectItem>
                          <SelectItem value="comprehensive">Comprehensive</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Expertise Level</Label>
                      <p className="text-sm text-muted-foreground">How technical should responses be?</p>
                      <Select
                        value={settings.expertiseLevel || "intermediate"}
                        onValueChange={(value) => updateSetting("expertiseLevel", value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="beginner">Beginner</SelectItem>
                          <SelectItem value="intermediate">Intermediate</SelectItem>
                          <SelectItem value="advanced">Advanced</SelectItem>
                          <SelectItem value="expert">Expert</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Model Configuration */}
                <div>
                  <h4 className="font-medium mb-4">Model Configuration</h4>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>System Prompt</Label>
                      <p className="text-sm text-muted-foreground">Advanced: Override the default system prompt</p>
                      <Textarea
                        value={settings.systemPrompt}
                        onChange={(e) => updateSetting("systemPrompt", e.target.value)}
                        placeholder="Enter a custom system prompt..."
                        rows={4}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Default Model</Label>
                      <Select
                        value={settings.defaultModel}
                        onValueChange={(value) => updateSetting("defaultModel", value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="gpt-4">GPT-4</SelectItem>
                          <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                          <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                          <SelectItem value="claude-3">Claude 3</SelectItem>
                          <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                          <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                          <SelectItem value="gemini-ultra">Gemini Ultra</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Max Tokens</Label>
                      <p className="text-sm text-muted-foreground">Maximum length of AI responses (1-8192)</p>
                      <Input
                        type="number"
                        value={settings.maxTokens}
                        onChange={(e) => updateSetting("maxTokens", Number.parseInt(e.target.value))}
                        min={1}
                        max={8192}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Temperature: {settings.temperature}</Label>
                      <p className="text-sm text-muted-foreground">
                        Controls randomness in responses (0 = focused, 2 = creative)
                      </p>
                      <Slider
                        value={[settings.temperature]}
                        onValueChange={(value) => updateSetting("temperature", value[0])}
                        max={2}
                        min={0}
                        step={0.1}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Focused</span>
                        <span>Creative</span>
                      </div>
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Display Preferences */}
                <div>
                  <h4 className="font-medium mb-4">Display Preferences</h4>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Show Timestamps</Label>
                        <p className="text-sm text-muted-foreground">Display message timestamps</p>
                      </div>
                      <Switch
                        checked={settings.showTimestamps}
                        onCheckedChange={(checked) => updateSetting("showTimestamps", checked)}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Show Word Count</Label>
                        <p className="text-sm text-muted-foreground">Display word count for messages</p>
                      </div>
                      <Switch
                        checked={settings.showWordCount || false}
                        onCheckedChange={(checked) => updateSetting("showWordCount", checked)}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Show Model Info</Label>
                        <p className="text-sm text-muted-foreground">Display which model generated each response</p>
                      </div>
                      <Switch
                        checked={settings.showModelInfo || false}
                        onCheckedChange={(checked) => updateSetting("showModelInfo", checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case "connections":
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium mb-4">Connections</h3>
              <div className="space-y-6">
                {/* Cloud Storage */}
                <div>
                  <h4 className="font-medium mb-4">Cloud Storage</h4>
                  <div className="space-y-4">
                    <div className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <Cloud className="h-5 w-5 text-blue-600" />
                          <div>
                            <Label className="text-base">Google Drive</Label>
                            <p className="text-sm text-muted-foreground">
                              Connect your Google Drive to save and access chat history
                            </p>
                          </div>
                        </div>
                        <div>
                          {settings.googleDriveConnected ? (
                            <div className="flex items-center gap-2">
                              <span className="text-sm text-green-600 font-medium">Connected</span>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  if (confirm("Are you sure you want to disconnect Google Drive?")) {
                                    updateSetting("googleDriveConnected", false)
                                    updateSetting("googleDriveEmail", "")
                                    logger.info("Disconnecting Google Drive", { component: "settings-dialog" })
                                  }
                                }}
                              >
                                Disconnect
                              </Button>
                            </div>
                          ) : (
                            <Button
                              onClick={() => {
                                // This would typically open Google OAuth flow
                                logger.info("Connecting to Google Drive", { component: "settings-dialog" })
                                // Simulate connection for demo
                                updateSetting("googleDriveConnected", true)
                                updateSetting("googleDriveEmail", "user@gmail.com")
                              }}
                            >
                              <Cloud className="h-4 w-4 mr-2" />
                              Connect
                            </Button>
                          )}
                        </div>
                      </div>
                      {settings.googleDriveConnected && (
                        <div className="mt-3 pt-3 border-t">
                          <p className="text-sm text-muted-foreground">
                            Connected as: <span className="font-medium">{settings.googleDriveEmail}</span>
                          </p>
                          <div className="mt-3 space-y-2">
                            <div className="flex items-center justify-between">
                              <div>
                                <Label className="text-sm">Auto-sync chat history</Label>
                                <p className="text-xs text-muted-foreground">
                                  Automatically backup chats to Google Drive
                                </p>
                              </div>
                              <Switch
                                checked={settings.googleDriveAutoSync || false}
                                onCheckedChange={(checked) => updateSetting("googleDriveAutoSync", checked)}
                              />
                            </div>
                            <div className="flex items-center justify-between">
                              <div>
                                <Label className="text-sm">Sync frequency</Label>
                              </div>
                              <Select
                                value={settings.googleDriveSyncFrequency || "daily"}
                                onValueChange={(value) => updateSetting("googleDriveSyncFrequency", value)}
                              >
                                <SelectTrigger className="w-32">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="realtime">Real-time</SelectItem>
                                  <SelectItem value="hourly">Hourly</SelectItem>
                                  <SelectItem value="daily">Daily</SelectItem>
                                  <SelectItem value="weekly">Weekly</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Integration Settings */}
                <div>
                  <h4 className="font-medium mb-4">Integration Settings</h4>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Share usage data with connected services</Label>
                        <p className="text-sm text-muted-foreground">
                          Help improve integrations by sharing anonymous usage data
                        </p>
                      </div>
                      <Switch
                        checked={settings.shareUsageWithIntegrations || false}
                        onCheckedChange={(checked) => updateSetting("shareUsageWithIntegrations", checked)}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Enable cross-platform sync</Label>
                        <p className="text-sm text-muted-foreground">
                          Sync settings and data across all connected platforms
                        </p>
                      </div>
                      <Switch
                        checked={settings.crossPlatformSync || true}
                        onCheckedChange={(checked) => updateSetting("crossPlatformSync", checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case "privacy":
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium mb-4">Privacy & Data</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Save Chat History</Label>
                    <p className="text-sm text-muted-foreground">Store conversations locally</p>
                  </div>
                  <Switch
                    checked={settings.saveHistory}
                    onCheckedChange={(checked) => updateSetting("saveHistory", checked)}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Share Analytics</Label>
                    <p className="text-sm text-muted-foreground">Help improve the service</p>
                  </div>
                  <Switch
                    checked={settings.shareAnalytics}
                    onCheckedChange={(checked) => updateSetting("shareAnalytics", checked)}
                  />
                </div>
                <Separator />
                <div className="space-y-4">
                  <h4 className="font-medium">API Keys</h4>{" "}
                  <div className="space-y-2">
                    <Label>OpenAI API Key</Label>
                    <Input
                      type="password"
                      value={settings.apiKeys?.openai || ""}
                      onChange={(e) => updateSetting("apiKeys", { ...settings.apiKeys, openai: e.target.value })}
                      placeholder="sk-..."
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Anthropic API Key</Label>
                    <Input
                      type="password"
                      value={settings.apiKeys?.anthropic || ""}
                      onChange={(e) => updateSetting("apiKeys", { ...settings.apiKeys, anthropic: e.target.value })}
                      placeholder="sk-ant-..."
                    />
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-4">Data Management</h4>
                  <div className="flex flex-wrap gap-4">
                    <Button onClick={exportSettings} variant="outline">
                      <Download className="h-4 w-4 mr-2" />
                      Export Settings
                    </Button>
                    <div>
                      <input
                        type="file"
                        accept=".json"
                        onChange={handleFileImport}
                        className="hidden"
                        id="import-settings"
                      />
                      <Button asChild variant="outline">
                        <label htmlFor="import-settings" className="cursor-pointer">
                          <Upload className="h-4 w-4 mr-2" />
                          Import Settings
                        </label>
                      </Button>
                    </div>
                    <Button onClick={resetSettings} variant="destructive">
                      <RotateCcw className="h-4 w-4 mr-2" />
                      Reset to Defaults
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="max-w-4xl h-[80vh] p-0 flex flex-col">
        <div className="flex flex-1 min-h-0">
          {/* Sidebar */}
          <div className="w-64 bg-muted/30 border-r">
            <DialogHeader className="p-6 pb-4">
              <DialogTitle className="text-xl font-semibold">Settings</DialogTitle>
            </DialogHeader>
            <nav className="px-3">
              {settingsSections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${
                    activeSection === section.id
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-background/50"
                  }`}
                >
                  <section.icon className="h-4 w-4" />
                  {section.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Content */}
          <div className="flex-1 flex flex-col min-h-0">
            <ScrollArea className="flex-1">
              <div className="p-6">{renderSectionContent()}</div>
            </ScrollArea>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
