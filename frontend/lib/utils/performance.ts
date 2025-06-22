import type { PerformanceMetrics } from "@/lib/types"

// Performance monitoring utilities
export class PerformanceMonitor {
  private static instance: PerformanceMonitor
  private metrics: Map<string, number[]> = new Map()

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor()
    }
    return PerformanceMonitor.instance
  }

  startTimer(label: string): () => number {
    const start = performance.now()

    return () => {
      const duration = performance.now() - start
      this.recordMetric(label, duration)
      return duration
    }
  }

  recordMetric(label: string, value: number): void {
    if (!this.metrics.has(label)) {
      this.metrics.set(label, [])
    }

    const values = this.metrics.get(label)!
    values.push(value)

    // Keep only last 100 measurements
    if (values.length > 100) {
      values.shift()
    }
  }

  getMetrics(label: string): PerformanceMetrics {
    const values = this.metrics.get(label) || []

    if (values.length === 0) {
      return { avg: 0, min: 0, max: 0, count: 0 }
    }

    const sum = values.reduce((a, b) => a + b, 0)
    const avg = sum / values.length
    const min = Math.min(...values)
    const max = Math.max(...values)

    return { avg, min, max, count: values.length }
  }

  getAllMetrics(): Record<string, { avg: number; min: number; max: number; count: number }> {
    const result: Record<string, { avg: number; min: number; max: number; count: number }> = {}

    this.metrics.forEach((values, label) => {
      result[label] = this.getMetrics(label)
    })

    return result
  }

  clearMetrics(label?: string): void {
    if (label) {
      this.metrics.delete(label)
    } else {
      this.metrics.clear()
    }
  }
}

// Memory usage monitoring
export const getMemoryUsage = (): { used: number; total: number; percentage: number } | null => {
  if (typeof performance === "undefined" || !(performance as any).memory) {
    return null
  }

  const memory = (performance as any).memory

  return {
    used: memory.usedJSHeapSize,
    total: memory.jsHeapSizeLimit,
    percentage: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100,
  }
}

// Network request monitoring
export const monitorFetch = (originalFetch: typeof fetch): typeof fetch => {
  return async (input: RequestInfo | URL, init?: RequestInit) => {
    const monitor = PerformanceMonitor.getInstance()
    const endTimer = monitor.startTimer(`fetch:${typeof input === "string" ? input : "request"}`)

    try {
      const response = await originalFetch(input, init)
      const duration = endTimer()

      // Record status code metrics
      monitor.recordMetric(`fetch:status:${response.status}`, 1)

      return response
    } catch (error) {
      endTimer()
      monitor.recordMetric("fetch:error", 1)
      throw error
    }
  }
}

// Component rendering performance
export const useRenderPerformance = (componentName: string): void => {
  if (typeof window === "undefined") return

  const monitor = PerformanceMonitor.getInstance()

  // Record component render
  monitor.recordMetric(`render:${componentName}`, 1)

  // Measure render time
  const endTimer = monitor.startTimer(`renderTime:${componentName}`)

  // Use requestAnimationFrame to measure after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      endTimer()
    })
  })
}

// Lazy loading utilities
export const lazyLoad = <T>(factory: () => Promise<T>): Promise<T> => {
  return new Promise((resolve) => {
    if (typeof window === "undefined") {
      // Server-side - load immediately
      resolve(factory())
    } else {
      // Client-side - use requestIdleCallback if available
      if ("requestIdleCallback" in window) {
        ;(window as any).requestIdleCallback(() => {
          resolve(factory())
        })
      } else {
        // Fallback to setTimeout
        setTimeout(() => {
          resolve(factory())
        }, 1)
      }
    }
  })
}

// Image optimization
export const optimizeImage = (url: string, width?: number, quality?: number): string => {
  if (!url) return url

  // If already using an image optimization service, return as is
  if (url.includes("imagedelivery.net") || url.includes("cloudinary.com")) {
    return url
  }

  // For Next.js Image Optimization
  const params = new URLSearchParams()

  if (width) {
    params.append("w", width.toString())
  }

  if (quality) {
    params.append("q", quality.toString())
  }

  const queryString = params.toString()
  return queryString ? `/_next/image?url=${encodeURIComponent(url)}&${queryString}` : url
}
