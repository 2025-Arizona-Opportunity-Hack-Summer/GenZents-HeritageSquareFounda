"use client"

import { useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useAuthStore } from "@/lib/stores/auth-store"
import { useToast } from "@/hooks/use-toast"

export function AuthForm() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [isResetMode, setIsResetMode] = useState(false)
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const router = useRouter()
  const searchParams = useSearchParams()
  const { toast } = useToast()
  const { signIn, resetPassword, loading } = useAuthStore()

  const redirectTo = searchParams.get("redirect") || "/"

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setIsLoading(true)

    try {
      if (isResetMode) {
        const result = await resetPassword(email)
        if (result.error) {
          setError(result.error)
        } else {
          toast({
            title: "Password reset email sent",
            description: "Check your email for password reset instructions.",
          })
          setIsResetMode(false)
          setEmail("")
        }
      } else {
        if (!email || !password) {
          setError("Please enter both email and password")
          return
        }

        const result = await signIn(email, password)
        if (result.error) {
          setError(result.error)
        } else {
          toast({
            title: "Signed in successfully",
            description: "Welcome back!",
          })
          router.push(redirectTo)
        }
      }
    } catch (err) {
      setError("An unexpected error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  const handleModeToggle = () => {
    setIsResetMode(!isResetMode)
    setError("")
    setPassword("")
  }

  return (
    <Card>
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl text-center">{isResetMode ? "Reset Password" : "Sign In"}</CardTitle>
        <CardDescription className="text-center">
          {isResetMode
            ? "Enter your email to receive a password reset link"
            : "Enter your credentials to access your account"}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={isLoading || loading}
            />
          </div>

          {!isResetMode && (
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                disabled={isLoading || loading}
              />
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Button type="submit" className="w-full" disabled={isLoading || loading}>
            {isLoading || loading ? "Please wait..." : isResetMode ? "Send Reset Email" : "Sign In"}
          </Button>
        </form>

        <div className="text-center">
          <Button variant="link" onClick={handleModeToggle} disabled={isLoading || loading} className="text-sm">
            {isResetMode ? "Back to sign in" : "Forgot your password?"}
          </Button>
        </div>

        {!isResetMode && (
          <div className="text-center text-sm text-muted-foreground">
            <p>Don&apos;t have an account? Contact your administrator for an invitation.</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
