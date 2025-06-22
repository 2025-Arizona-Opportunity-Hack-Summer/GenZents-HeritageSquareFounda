import { getSupabaseClient } from "@/lib/supabase/server"
import { logger } from "@/lib/utils/logger"
import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const supabase = getSupabaseClient()
    const { message, chatId } = await request.json()

    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser()

    if (authError || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Get the session token
    const { data: sessionData } = await supabase.auth.getSession()
    const token = sessionData.session?.access_token

    if (!token) {
      return NextResponse.json({ error: "No session token" }, { status: 401 })
    }

    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000"
    const res = await fetch(`${backendUrl}/api/messages/chat/${chatId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        role: "user",
        content: message,
      }),
    })

    if (!res.ok) {
      const errorText = await res.text()
      let errorData
      try {
        errorData = JSON.parse(errorText)
      } catch {
        errorData = { detail: errorText }
      }
      return NextResponse.json(
        {
          error: errorData.detail || "Failed to process message",
        },
        { status: res.status }
      )
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    logger.error("Chat API error", error as Error, { component: "chat-api" })
    return NextResponse.json(
      {
        error: "Internal server error",
      },
      { status: 500 }
    )
  }
}
