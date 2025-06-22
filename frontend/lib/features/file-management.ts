import { supabase } from "@/lib/supabase/client"
import type { Attachment, VirusScanResult, UploadOptions } from "@/lib/types"

// Remove duplicate interfaces - now using centralized types

export class FileManager {
  private static readonly DEFAULT_MAX_SIZE = 100 * 1024 * 1024 // 100MB
  private static readonly DEFAULT_ALLOWED_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/json",
    "text/csv",
    "audio/mpeg",
    "audio/wav",
    "audio/ogg",
    "video/mp4",
    "video/webm",
    "video/ogg",
  ]

  static async uploadFile(file: File, chatId: string, options: UploadOptions = {}): Promise<Attachment> {
    const {
      maxSize = this.DEFAULT_MAX_SIZE,
      allowedTypes = this.DEFAULT_ALLOWED_TYPES,
      generateThumbnail = true,
      virusScan = true,
      encrypt = false,
    } = options

    // Validate file
    this.validateFile(file, maxSize, allowedTypes)

    // Generate unique filename
    const fileExtension = file.name.split(".").pop()
    const fileName = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}.${fileExtension}`
    const filePath = `chats/${chatId}/attachments/${fileName}`

    // Upload to Supabase Storage
    const { data: uploadData, error: uploadError } = await supabase.storage.from("attachments").upload(filePath, file, {
      cacheControl: "3600",
      upsert: false,
    })

    if (uploadError) {
      throw new Error(`Upload failed: ${uploadError.message}`)
    }

    // Get public URL
    const { data: urlData } = supabase.storage.from("attachments").getPublicUrl(filePath)

    // Generate thumbnail if needed
    let thumbnailUrl: string | undefined
    if (generateThumbnail && file.type.startsWith("image/")) {
      thumbnailUrl = await this.generateThumbnail(file, filePath)
    }
    thumbnailUrl = await this.generateThumbnail(file, filePath)

    // Perform virus scan if enabled
    let virusScanResult: VirusScanResult = {
      scanned: false,
      clean: true,
      scanned_at: new Date().toISOString(),
    }

    if (virusScan) {
      virusScanResult = await this.performVirusScan(file)
      if (!virusScanResult.clean) {
        // Delete uploaded file if virus detected
        await supabase.storage.from("attachments").remove([filePath])
        throw new Error("File failed virus scan")
      }
    }

    // Calculate file metadata
    const metadata = await this.extractMetadata(file)

    // Create attachment record
    const attachment: Partial<Attachment> = {
      name: file.name,
      type: file.type,
      size: file.size,
      url: urlData.publicUrl,
      thumbnail: thumbnailUrl,
      metadata,
      virus_scan: virusScanResult,
      created_at: new Date().toISOString(),
    }

    const { data: attachmentData, error: attachmentError } = await supabase
      .from("attachments")
      .insert(attachment)
      .select()
      .single()

    if (attachmentError) {
      throw new Error(`Failed to create attachment record: ${attachmentError.message}`)
    }

    return attachmentData
  }

  private static validateFile(file: File, maxSize: number, allowedTypes: string[]) {
    if (file.size > maxSize) {
      throw new Error(`File size exceeds maximum allowed size of ${maxSize / 1024 / 1024}MB`)
    }

    if (!allowedTypes.includes(file.type)) {
      throw new Error(`File type ${file.type} is not allowed`)
    }
  }

  private static async generateThumbnail(file: File, filePath: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")
      const img = new Image()

      img.onload = async () => {
        // Calculate thumbnail dimensions
        const maxSize = 200
        let { width, height } = img

        if (width > height) {
          if (width > maxSize) {
            height = (height * maxSize) / width
            width = maxSize
          }
        } else {
          if (height > maxSize) {
            width = (width * maxSize) / height
            height = maxSize
          }
        }

        canvas.width = width
        canvas.height = height
        ctx?.drawImage(img, 0, 0, width, height)

        canvas.toBlob(
          async (blob) => {
            if (!blob) {
              reject(new Error("Failed to generate thumbnail"))
              return
            }

            const thumbnailPath = filePath.replace(/\.[^/.]+$/, "_thumb.jpg")

            const { data, error } = await supabase.storage.from("attachments").upload(thumbnailPath, blob)

            if (error) {
              reject(error)
              return
            }

            const { data: urlData } = supabase.storage.from("attachments").getPublicUrl(thumbnailPath)

            resolve(urlData.publicUrl)
          },
          "image/jpeg",
          0.8
        )
      }

      img.onerror = () => reject(new Error("Failed to load image for thumbnail"))
      img.src = URL.createObjectURL(file)
    })
  }

  private static async performVirusScan(file: File): Promise<VirusScanResult> {
    // Implementation would integrate with virus scanning service
    // For now, return a mock result
    return {
      scanned: true,
      clean: true,
      scanned_at: new Date().toISOString(),
    }
  }

  private static async extractMetadata(file: File): Promise<any> {
    const metadata: any = {
      checksum: await this.calculateChecksum(file),
    }

    if (file.type.startsWith("image/")) {
      const img = new Image()
      img.src = URL.createObjectURL(file)
      await new Promise((resolve) => (img.onload = resolve))
      metadata.width = img.width
      metadata.height = img.height
    }

    if (file.type.startsWith("video/")) {
      const video = document.createElement("video")
      video.src = URL.createObjectURL(file)
      await new Promise((resolve) => (video.onloadedmetadata = resolve))
      metadata.width = video.videoWidth
      metadata.height = video.videoHeight
      metadata.duration = video.duration
    }

    if (file.type.startsWith("audio/")) {
      const audio = document.createElement("audio")
      audio.src = URL.createObjectURL(file)
      await new Promise((resolve) => (audio.onloadedmetadata = resolve))
      metadata.duration = audio.duration
    }

    return metadata
  }

  private static async calculateChecksum(file: File): Promise<string> {
    const buffer = await file.arrayBuffer()
    const hashBuffer = await crypto.subtle.digest("SHA-256", buffer)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")
  }

  static async deleteFile(attachmentId: string): Promise<void> {
    const { data: attachment, error: fetchError } = await supabase
      .from("attachments")
      .select("url, thumbnail")
      .eq("id", attachmentId)
      .single()

    if (fetchError || !attachment) {
      throw new Error("Attachment not found")
    }

    // Extract file path from URL
    const url = new URL(attachment.url)
    const filePath = url.pathname.split("/").slice(-3).join("/")

    // Delete from storage
    const filesToDelete = [filePath]
    if (attachment.thumbnail) {
      const thumbnailUrl = new URL(attachment.thumbnail)
      const thumbnailPath = thumbnailUrl.pathname.split("/").slice(-3).join("/")
      filesToDelete.push(thumbnailPath)
    }

    await supabase.storage.from("attachments").remove(filesToDelete)

    // Delete attachment record
    const { error: deleteError } = await supabase.from("attachments").delete().eq("id", attachmentId)

    if (deleteError) {
      throw new Error(`Failed to delete attachment record: ${deleteError.message}`)
    }
  }

  static async getAttachment(attachmentId: string): Promise<Attachment | null> {
    const { data, error } = await supabase.from("attachments").select("*").eq("id", attachmentId).single()

    if (error) return null
    return data
  }

  static async getAttachmentsByChat(chatId: string): Promise<Attachment[]> {
    const { data, error } = await supabase
      .from("attachments")
      .select("*")
      .eq("chat_id", chatId)
      .order("created_at", { ascending: false })

    if (error) return []
    return data || []
  }
}
