import type { KeyboardShortcutOptions } from "@/lib/types"

// Focus management
export const focusElement = (selector: string): void => {
  const element = document.querySelector(selector) as HTMLElement
  if (element) {
    element.focus()
  }
}

export const trapFocus = (element: HTMLElement): (() => void) => {
  const focusableElements = element.querySelectorAll(
    'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
  ) as NodeListOf<HTMLElement>

  if (focusableElements.length === 0) return () => {}

  const firstElement = focusableElements[0]
  const lastElement = focusableElements[focusableElements.length - 1]

  const handleTabKey = (e: KeyboardEvent) => {
    if (e.key !== "Tab") return

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault()
        lastElement.focus()
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault()
        firstElement.focus()
      }
    }
  }

  element.addEventListener("keydown", handleTabKey)
  firstElement.focus()

  return () => {
    element.removeEventListener("keydown", handleTabKey)
  }
}

// Keyboard navigation
export const registerKeyboardShortcut = (
  key: string,
  callback: (e: KeyboardEvent) => void,
  options: KeyboardShortcutOptions = {}
): (() => void) => {
  const handler = (e: KeyboardEvent) => {
    if (
      e.key.toLowerCase() === key.toLowerCase() &&
      !!options.ctrl === e.ctrlKey &&
      !!options.alt === e.altKey &&
      !!options.shift === e.shiftKey &&
      !!options.meta === e.metaKey
    ) {
      callback(e)
    }
  }

  window.addEventListener("keydown", handler)
  return () => window.removeEventListener("keydown", handler)
}

// Screen reader announcements
export const announce = (message: string, priority: "polite" | "assertive" = "polite"): void => {
  let announcer = document.getElementById("announcer")

  if (!announcer) {
    announcer = document.createElement("div")
    announcer.id = "announcer"
    announcer.setAttribute("aria-live", priority)
    announcer.setAttribute("aria-atomic", "true")
    announcer.style.position = "absolute"
    announcer.style.width = "1px"
    announcer.style.height = "1px"
    announcer.style.padding = "0"
    announcer.style.overflow = "hidden"
    announcer.style.clip = "rect(0, 0, 0, 0)"
    announcer.style.whiteSpace = "nowrap"
    announcer.style.border = "0"
    document.body.appendChild(announcer)
  } else {
    announcer.setAttribute("aria-live", priority)
  }

  // Clear the announcer
  announcer.textContent = ""

  // Set the message after a brief delay to ensure it's announced
  setTimeout(() => {
    announcer!.textContent = message
  }, 50)
}

// Color contrast
export const getContrastRatio = (foreground: string, background: string): number => {
  const getLuminance = (color: string): number => {
    // Convert hex to RGB
    let r, g, b

    if (color.startsWith("#")) {
      const hex = color.slice(1)
      r = Number.parseInt(hex.slice(0, 2), 16) / 255
      g = Number.parseInt(hex.slice(2, 4), 16) / 255
      b = Number.parseInt(hex.slice(4, 6), 16) / 255
    } else if (color.startsWith("rgb")) {
      const match = color.match(/(\d+),\s*(\d+),\s*(\d+)/)
      if (!match) return 0
      r = Number.parseInt(match[1]) / 255
      g = Number.parseInt(match[2]) / 255
      b = Number.parseInt(match[3]) / 255
    } else {
      return 0
    }

    // Calculate luminance
    const toLinear = (val: number) => {
      return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4)
    }

    r = toLinear(r)
    g = toLinear(g)
    b = toLinear(b)

    return 0.2126 * r + 0.7152 * g + 0.0722 * b
  }

  const luminance1 = getLuminance(foreground)
  const luminance2 = getLuminance(background)

  const lighter = Math.max(luminance1, luminance2)
  const darker = Math.min(luminance1, luminance2)

  return (lighter + 0.05) / (darker + 0.05)
}

export const isAccessibleColor = (foreground: string, background: string, level: "AA" | "AAA" = "AA"): boolean => {
  const ratio = getContrastRatio(foreground, background)
  return level === "AA" ? ratio >= 4.5 : ratio >= 7
}

// Reduced motion
export const prefersReducedMotion = (): boolean => {
  if (typeof window === "undefined") return false
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
}

export const getReducedMotionStyle = (normalStyle: string, reducedStyle: string): string => {
  return prefersReducedMotion() ? reducedStyle : normalStyle
}

// High contrast
export const prefersHighContrast = (): boolean => {
  if (typeof window === "undefined") return false
  return window.matchMedia("(prefers-contrast: more)").matches
}

// Language detection
export const getBrowserLanguage = (): string => {
  if (typeof navigator === "undefined") return "en"
  return navigator.language || "en"
}
