enum LogLevel {
  ERROR = "error",
  WARN = "warn",
  INFO = "info",
  DEBUG = "debug",
}

interface LogContext {
  component?: string
  action?: string
  userId?: string
  [key: string]: any
}

class Logger {
  private isDev = process.env.NEXT_PUBLIC_NODE_ENV === "development"
  private isClient = typeof window !== "undefined"

  private formatMessage(level: LogLevel, message: string, context?: LogContext): string {
    const timestamp = new Date().toISOString()
    const contextStr = context ? ` [${JSON.stringify(context)}]` : ""
    return `[${timestamp}] ${level.toUpperCase()}: ${message}${contextStr}`
  }

  private shouldLog(level: LogLevel): boolean {
    if (!this.isClient) return false
    return this.isDev || level === LogLevel.ERROR
  }

  error(message: string, error?: Error, context?: LogContext): void {
    if (!this.shouldLog(LogLevel.ERROR)) return

    const formattedMessage = this.formatMessage(LogLevel.ERROR, message, context)
    console.error(formattedMessage, error || "")
  }

  warn(message: string, context?: LogContext): void {
    if (!this.shouldLog(LogLevel.WARN)) return

    const formattedMessage = this.formatMessage(LogLevel.WARN, message, context)
    console.warn(formattedMessage)
  }

  info(message: string, context?: LogContext): void {
    if (!this.shouldLog(LogLevel.INFO)) return

    const formattedMessage = this.formatMessage(LogLevel.INFO, message, context)
    console.info(formattedMessage)
  }

  debug(message: string, context?: LogContext): void {
    if (!this.shouldLog(LogLevel.DEBUG)) return

    const formattedMessage = this.formatMessage(LogLevel.DEBUG, message, context)
    console.debug(formattedMessage)
  }
}

export const logger = new Logger()
