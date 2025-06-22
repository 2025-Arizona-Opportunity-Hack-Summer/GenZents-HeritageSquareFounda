"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "@/hooks/theme-provider"
import { Button } from "@/components/ui/button"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  const toggleTheme = () => {
    if (theme === "light") {
      setTheme("dark")
    } else {
      setTheme("light")
    }
  }

  const getIcon = () => {
    if (theme === "light") {
      return <Sun className="h-4 w-4" />
    } else {
      return <Moon className="h-4 w-4" />
    }
  }

  return (
    <Button variant="ghost" size="icon" className="h-9 w-9" onClick={toggleTheme}>
      {getIcon()}
      <span className="sr-only">Toggle theme</span>
    </Button>
  )
}
