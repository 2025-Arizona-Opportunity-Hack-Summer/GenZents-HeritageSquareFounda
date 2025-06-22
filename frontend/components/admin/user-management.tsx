"use client"

import { useState, useEffect } from "react"
import { useAdminStore } from "@/lib/stores/admin-store"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Loader2, UserPlus, MoreHorizontal, Trash2, UserCheck, UserX, Settings } from "lucide-react"
import type { UserPermission, UserStatus } from "@/lib/types/database"

const PERMISSION_OPTIONS: { value: UserPermission; label: string; description: string }[] = [
  { value: "ai_chat", label: "AI Chat", description: "Can chat with AI assistants" },
  { value: "file_organization", label: "File Organization", description: "Can organize files and folders" },
  { value: "version_history", label: "Version History", description: "Can manage version history" },
  { value: "context_management", label: "Context Management", description: "Can manage context" },
  { value: "tools_access", label: "Tools Access", description: "Can access additional tools" },
]

export function AdminUserManagement() {
  const { users, loading, fetchUsers, inviteUser, updateUser, deleteUser, toggleUserStatus, updateUserPermissions } =
    useAdminStore()

  const [showInviteDialog, setShowInviteDialog] = useState(false)
  const [inviteEmail, setInviteEmail] = useState("")
  const [inviteFullName, setInviteFullName] = useState("")
  const [invitePermissions, setInvitePermissions] = useState<UserPermission[]>([])
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

  const handleInviteUser = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setSuccess("")

    if (!inviteEmail || !inviteFullName) {
      setError("Please fill in all required fields")
      return
    }

    if (invitePermissions.length === 0) {
      setError("Please select at least one permission")
      return
    }

    const result = await inviteUser(inviteEmail, inviteFullName, invitePermissions)
    if (result.error) {
      setError(result.error)
    } else {
      setSuccess("User invitation sent successfully!")
      setShowInviteDialog(false)
      setInviteEmail("")
      setInviteFullName("")
      setInvitePermissions([])
      setTimeout(() => setSuccess(""), 3000)
    }
  }

  const handleDeleteUser = async (userId: string) => {
    if (!confirm("Are you sure you want to delete this user? This action cannot be undone.")) {
      return
    }

    const result = await deleteUser(userId)
    if (result.error) {
      setError(result.error)
      setTimeout(() => setError(""), 3000)
    }
  }

  const handleToggleStatus = async (userId: string, currentStatus: UserStatus) => {
    const newStatus: UserStatus = currentStatus === "active" ? "paused" : "active"
    const result = await toggleUserStatus(userId, newStatus)
    if (result.error) {
      setError(result.error)
      setTimeout(() => setError(""), 3000)
    }
  }

  const handlePermissionChange = (permission: UserPermission, checked: boolean) => {
    if (checked) {
      setInvitePermissions([...invitePermissions, permission])
    } else {
      setInvitePermissions(invitePermissions.filter((p) => p !== permission))
    }
  }

  const handleUserPermissionUpdate = async (userId: string, permissions: UserPermission[]) => {
    const result = await updateUserPermissions(userId, permissions)
    if (result.error) {
      setError(result.error)
      setTimeout(() => setError(""), 3000)
    }
  }

  const getStatusBadge = (status: UserStatus) => {
    switch (status) {
      case "active":
        return (
          <Badge variant="default" className="bg-green-100 text-green-800">
            Active
          </Badge>
        )
      case "paused":
        return (
          <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">
            Paused
          </Badge>
        )
      case "pending_invitation":
        return (
          <Badge variant="outline" className="bg-blue-100 text-blue-800">
            Pending
          </Badge>
        )
      default:
        return <Badge variant="outline">Unknown</Badge>
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">User Management</h1>
          <p className="text-muted-foreground">Manage users, permissions, and access control</p>
        </div>

        <Dialog open={showInviteDialog} onOpenChange={setShowInviteDialog}>
          <DialogTrigger asChild>
            <Button>
              <UserPlus className="mr-2 h-4 w-4" />
              Invite User
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Invite New User</DialogTitle>
              <DialogDescription>Send an invitation to a new user with specific permissions</DialogDescription>
            </DialogHeader>

            <form onSubmit={handleInviteUser} className="space-y-4">
              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="user@example.com"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="fullName">Full Name</Label>
                <Input
                  id="fullName"
                  type="text"
                  placeholder="John Doe"
                  value={inviteFullName}
                  onChange={(e) => setInviteFullName(e.target.value)}
                  required
                />
              </div>

              <div className="space-y-3">
                <Label>Permissions</Label>
                <div className="space-y-3">
                  {PERMISSION_OPTIONS.map((permission) => (
                    <div key={permission.value} className="flex items-start space-x-3">
                      <Checkbox
                        id={permission.value}
                        checked={invitePermissions.includes(permission.value)}
                        onCheckedChange={(checked) => handlePermissionChange(permission.value, checked as boolean)}
                      />
                      <div className="grid gap-1.5 leading-none">
                        <Label
                          htmlFor={permission.value}
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                        >
                          {permission.label}
                        </Label>
                        <p className="text-xs text-muted-foreground">{permission.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setShowInviteDialog(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={loading}>
                  {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Send Invitation
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {success && (
        <Alert>
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Users ({users.length})</CardTitle>
          <CardDescription>Manage user accounts, permissions, and access status</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>User</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Permissions</TableHead>
                  <TableHead>Last Active</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users.map((user) => (
                  <TableRow key={user.id}>
                    <TableCell>
                      <div>
                        <div className="font-medium">{user.full_name || "No name"}</div>
                        <div className="text-sm text-muted-foreground">{user.email}</div>
                        {user.is_admin && (
                          <Badge variant="outline" className="text-xs mt-1">
                            Admin
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>{getStatusBadge(user.status)}</TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {user.permissions?.map((permission) => (
                          <Badge key={permission} variant="secondary" className="text-xs">
                            {PERMISSION_OPTIONS.find((p) => p.value === permission)?.label || permission}
                          </Badge>
                        ))}
                      </div>
                    </TableCell>
                    <TableCell>
                      {user.last_active ? new Date(user.last_active).toLocaleDateString() : "Never"}
                    </TableCell>
                    <TableCell className="text-right">
                      {!user.is_admin && (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" className="h-8 w-8 p-0">
                              <span className="sr-only">Open menu</span>
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => handleToggleStatus(user.id, user.status)}>
                              {user.status === "active" ? (
                                <>
                                  <UserX className="mr-2 h-4 w-4" />
                                  Pause Access
                                </>
                              ) : (
                                <>
                                  <UserCheck className="mr-2 h-4 w-4" />
                                  Resume Access
                                </>
                              )}
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleDeleteUser(user.id)} className="text-red-600">
                              <Trash2 className="mr-2 h-4 w-4" />
                              Delete User
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
