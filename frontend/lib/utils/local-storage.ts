import { encryptMessage, decryptMessage } from "./security"

interface SecureStorageOptions {
  encrypt?: boolean
  expiryHours?: number
}

interface StorageItem {
  value: any
  encrypted: boolean
  timestamp: number
  expiry?: number
}

const STORAGE_KEY_PREFIX = "archyx_"
const DEFAULT_ENCRYPTION_KEY = "default_key_should_be_replaced_in_production"

export class SecureStorage {
  private static getKey(key: string): string {
    return `${STORAGE_KEY_PREFIX}${key}`
  }

  static async setItem(
    key: string, 
    value: any, 
    options: SecureStorageOptions = {}
  ): Promise<void> {
    try {
      const { encrypt = false, expiryHours } = options
      const timestamp = Date.now()
      
      const item: StorageItem = {
        value: encrypt ? await encryptMessage(JSON.stringify(value), DEFAULT_ENCRYPTION_KEY) : value,
        encrypted: encrypt,
        timestamp,
        expiry: expiryHours ? timestamp + (expiryHours * 60 * 60 * 1000) : undefined
      }

      localStorage.setItem(this.getKey(key), JSON.stringify(item))
    } catch (error) {
      console.error("Failed to set secure storage item:", error)
    }
  }

  static async getItem<T>(key: string): Promise<T | null> {
    try {
      const stored = localStorage.getItem(this.getKey(key))
      if (!stored) return null

      const item: StorageItem = JSON.parse(stored)
      
      // Check expiry
      if (item.expiry && Date.now() > item.expiry) {
        this.removeItem(key)
        return null
      }

      if (item.encrypted) {
        const decryptedValue = await decryptMessage(item.value, DEFAULT_ENCRYPTION_KEY)
        return JSON.parse(decryptedValue)
      }

      return item.value
    } catch (error) {
      console.error("Failed to get secure storage item:", error)
      return null
    }
  }

  static removeItem(key: string): void {
    localStorage.removeItem(this.getKey(key))
  }

  static clear(): void {
    const keys = Object.keys(localStorage)
    keys.forEach(key => {
      if (key.startsWith(STORAGE_KEY_PREFIX)) {
        localStorage.removeItem(key)
      }
    })
  }

  static async clearExpired(): Promise<void> {
    const keys = Object.keys(localStorage)
    const now = Date.now()
    
    for (const key of keys) {
      if (key.startsWith(STORAGE_KEY_PREFIX)) {
        try {
          const stored = localStorage.getItem(key)
          if (stored) {
            const item: StorageItem = JSON.parse(stored)
            if (item.expiry && now > item.expiry) {
              localStorage.removeItem(key)
            }
          }
        } catch (error) {
          // Remove corrupted items
          localStorage.removeItem(key)
        }
      }
    }
  }
}

// Auto-cleanup expired items on page load
if (typeof window !== 'undefined') {
  SecureStorage.clearExpired()
}
