"use client"

import { Suspense } from "react"
import { AuthForm } from "@/components/auth/auth-form"

function AuthContent() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4 py-12">
      <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px]">
        <div className="flex flex-col space-y-2 text-center">
          <h1 className="text-3xl font-bold tracking-tight">Archyx AI</h1>
          <p className="text-sm text-muted-foreground">Get started with your account</p>
        </div>
        <AuthForm />
      </div>
    </div>
  )
}

export default function AuthPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <AuthContent />
    </Suspense>
  )
}
