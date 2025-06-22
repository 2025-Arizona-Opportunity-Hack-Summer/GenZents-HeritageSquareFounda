import type { RateLimitOptions, RateLimitStore, RateLimitResult } from "@/lib/types"

const store: RateLimitStore = {}

// Clean up expired entries periodically
setInterval(() => {
  const now = Date.now()
  Object.keys(store).forEach(key => {
    if (store[key] && now > store[key].resetTime) {
      delete store[key]
    }
  })
}, 60000) // Clean up every minute

export const rateLimit = (options: RateLimitOptions) => {
  return (identifier: string): RateLimitResult => {
    const now = Date.now()
    const key = identifier

    // Clean up expired entries
    if (store[key] && now > store[key].resetTime) {
      delete store[key]
    }

    // Initialize or get current count
    if (!store[key]) {
      store[key] = {
        count: 0,
        resetTime: now + options.windowMs,
      }
    }

    store[key].count++

    const allowed = store[key].count <= options.max
    const remaining = Math.max(0, options.max - store[key].count)

    return {
      allowed,
      remaining,
      resetTime: store[key].resetTime,
    }
  }
}

// Usage example:
// const limiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 100, message: "Too many requests", standardHeaders: true, legacyHeaders: false })
// const result = limiter(userIP)
