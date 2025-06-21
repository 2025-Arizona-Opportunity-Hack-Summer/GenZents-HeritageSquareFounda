export type UserPermission =
  | "ai_chat"
  | "file_organization"
  | "version_history"
  | "context_management"
  | "tools_access"
  | "admin_access"

export type UserStatus = "active" | "paused" | "pending_invitation"

// Database types that match the Supabase schema
export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string
          email: string
          full_name: string | null
          avatar_url: string | null
          created_at: string
          updated_at: string
          theme: "light" | "dark" | "system"
          language: string
          timezone: string
          is_admin: boolean
          status: UserStatus
          permissions: UserPermission[]
          last_sign_in: string | null
          invitation_token: string | null
          invitation_expires_at: string | null
          subscription_plan: "free" | "pro" | "enterprise"
          subscription_features: string[]
          billing_amount: number
          billing_currency: string
          billing_interval: "monthly" | "yearly" | null
          next_billing: string | null
          messages_per_day: number
          tokens_per_month: number
          file_uploads: number
          custom_models: number
          messages_count: number
          tokens_used: number
          files_uploaded: number
          last_active: string
        }
        Insert: {
          id: string
          email: string
          full_name?: string | null
          avatar_url?: string | null
          theme?: "light" | "dark" | "system"
          language?: string
          timezone?: string
          is_admin?: boolean
          status?: UserStatus
          permissions?: UserPermission[]
          last_sign_in?: string | null
          invitation_token?: string | null
          invitation_expires_at?: string | null
          subscription_plan?: "free" | "pro" | "enterprise"
          subscription_features?: string[]
          billing_amount?: number
          billing_currency?: string
          billing_interval?: "monthly" | "yearly" | null
          next_billing?: string | null
          messages_per_day?: number
          tokens_per_month?: number
          file_uploads?: number
          custom_models?: number
          messages_count?: number
          tokens_used?: number
          files_uploaded?: number
          last_active?: string
        }
        Update: {
          id?: string
          email?: string
          full_name?: string | null
          avatar_url?: string | null
          theme?: "light" | "dark" | "system"
          language?: string
          timezone?: string
          is_admin?: boolean
          status?: UserStatus
          permissions?: UserPermission[]
          last_sign_in?: string | null
          invitation_token?: string | null
          invitation_expires_at?: string | null
          subscription_plan?: "free" | "pro" | "enterprise"
          subscription_features?: string[]
          billing_amount?: number
          billing_currency?: string
          billing_interval?: "monthly" | "yearly" | null
          next_billing?: string | null
          messages_per_day?: number
          tokens_per_month?: number
          file_uploads?: number
          custom_models?: number
          messages_count?: number
          tokens_used?: number
          files_uploaded?: number
          last_active?: string
          updated_at?: string
        }
      }
      user_settings: {
        Row: {
          id: string
          user_id: string
          email_notifications: boolean
          push_notifications: boolean
          desktop_notifications: boolean
          sound_notifications: boolean
          mention_notifications: boolean
          reply_notifications: boolean
          profile_visibility: "public" | "private" | "friends"
          data_sharing: boolean
          analytics: boolean
          marketing: boolean
          high_contrast: boolean
          reduced_motion: boolean
          screen_reader: boolean
          font_size: "small" | "medium" | "large" | "xl"
          keyboard_navigation: boolean
          default_model: string
          temperature: number
          system_prompt: string
          max_tokens: number
          custom_instructions: string
          communication_style: "professional" | "casual" | "friendly" | "balanced" | "technical"
          response_length: "concise" | "balanced" | "detailed" | "comprehensive"
          expertise_level: "beginner" | "intermediate" | "advanced" | "expert"
          show_timestamps: boolean
          show_word_count: boolean
          show_model_info: boolean
          show_token_count: boolean
          show_avatars: boolean
          compact_mode: boolean
          full_screen_mode: boolean
          auto_save: boolean
          auto_save_interval: number
          encrypt_messages: boolean
          retention_days: number
          created_at: string
          updated_at: string
        }
        Insert: {
          user_id: string
          email_notifications?: boolean
          push_notifications?: boolean
          desktop_notifications?: boolean
          sound_notifications?: boolean
          mention_notifications?: boolean
          reply_notifications?: boolean
          profile_visibility?: "public" | "private" | "friends"
          data_sharing?: boolean
          analytics?: boolean
          marketing?: boolean
          high_contrast?: boolean
          reduced_motion?: boolean
          screen_reader?: boolean
          font_size?: "small" | "medium" | "large" | "xl"
          keyboard_navigation?: boolean
          default_model?: string
          temperature?: number
          system_prompt?: string
          max_tokens?: number
          custom_instructions?: string
          communication_style?: "professional" | "casual" | "friendly" | "balanced" | "technical"
          response_length?: "concise" | "balanced" | "detailed" | "comprehensive"
          expertise_level?: "beginner" | "intermediate" | "advanced" | "expert"
          show_timestamps?: boolean
          show_word_count?: boolean
          show_model_info?: boolean
          show_token_count?: boolean
          show_avatars?: boolean
          compact_mode?: boolean
          full_screen_mode?: boolean
          auto_save?: boolean
          auto_save_interval?: number
          encrypt_messages?: boolean
          retention_days?: number
        }
        Update: {
          user_id?: string
          email_notifications?: boolean
          push_notifications?: boolean
          desktop_notifications?: boolean
          sound_notifications?: boolean
          mention_notifications?: boolean
          reply_notifications?: boolean
          profile_visibility?: "public" | "private" | "friends"
          data_sharing?: boolean
          analytics?: boolean
          marketing?: boolean
          high_contrast?: boolean
          reduced_motion?: boolean
          screen_reader?: boolean
          font_size?: "small" | "medium" | "large" | "xl"
          keyboard_navigation?: boolean
          default_model?: string
          temperature?: number
          system_prompt?: string
          max_tokens?: number
          custom_instructions?: string
          communication_style?: "professional" | "casual" | "friendly" | "balanced" | "technical"
          response_length?: "concise" | "balanced" | "detailed" | "comprehensive"
          expertise_level?: "beginner" | "intermediate" | "advanced" | "expert"
          show_timestamps?: boolean
          show_word_count?: boolean
          show_model_info?: boolean
          show_token_count?: boolean
          show_avatars?: boolean
          compact_mode?: boolean
          full_screen_mode?: boolean
          auto_save?: boolean
          auto_save_interval?: number
          encrypt_messages?: boolean
          retention_days?: number
          updated_at?: string
        }
      }
      ai_models: {
        Row: {
          id: string
          name: string
          provider: string
          type: "text" | "image" | "audio" | "video" | "multimodal"
          capabilities: any
          pricing: any
          limits: any
          status: "active" | "deprecated" | "beta"
          description: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          name: string
          provider: string
          type: "text" | "image" | "audio" | "video" | "multimodal"
          capabilities?: any
          pricing?: any
          limits?: any
          status?: "active" | "deprecated" | "beta"
          description?: string | null
        }
        Update: {
          name?: string
          provider?: string
          type?: "text" | "image" | "audio" | "video" | "multimodal"
          capabilities?: any
          pricing?: any
          limits?: any
          status?: "active" | "deprecated" | "beta"
          description?: string | null
          updated_at?: string
        }
      }
      chats: {
        Row: {
          id: string
          title: string
          description: string | null
          user_id: string
          category: string
          status: "active" | "archived" | "deleted"
          visibility: "private" | "shared" | "public"
          is_template: boolean
          bookmarked: boolean
          archived: boolean
          shared: boolean
          tags: string[]
          model: string
          temperature: number
          max_tokens: number
          system_prompt: string
          context_window: number
          streaming: boolean
          auto_save: boolean
          encryption: boolean
          retention_days: number
          total_messages: number
          total_tokens: number
          average_response_time: number
          last_activity: string
          participants: string[]
          language: string
          domain: string
          version: number
          parent_version: string | null
          branch: string
          created_at: string
          updated_at: string
        }
        Insert: {
          title?: string
          description?: string | null
          user_id: string
          category?: string
          status?: "active" | "archived" | "deleted"
          visibility?: "private" | "shared" | "public"
          is_template?: boolean
          bookmarked?: boolean
          archived?: boolean
          shared?: boolean
          tags?: string[]
          model?: string
          temperature?: number
          max_tokens?: number
          system_prompt?: string
          context_window?: number
          streaming?: boolean
          auto_save?: boolean
          encryption?: boolean
          retention_days?: number
          total_messages?: number
          total_tokens?: number
          average_response_time?: number
          last_activity?: string
          participants?: string[]
          language?: string
          domain?: string
          version?: number
          parent_version?: string | null
          branch?: string
        }
        Update: {
          title?: string
          description?: string | null
          user_id?: string
          category?: string
          status?: "active" | "archived" | "deleted"
          visibility?: "private" | "shared" | "public"
          is_template?: boolean
          bookmarked?: boolean
          archived?: boolean
          shared?: boolean
          tags?: string[]
          model?: string
          temperature?: number
          max_tokens?: number
          system_prompt?: string
          context_window?: number
          streaming?: boolean
          auto_save?: boolean
          encryption?: boolean
          retention_days?: number
          total_messages?: number
          total_tokens?: number
          average_response_time?: number
          last_activity?: string
          participants?: string[]
          language?: string
          domain?: string
          version?: number
          parent_version?: string | null
          branch?: string
          updated_at?: string
        }
      }
      messages: {
        Row: {
          id: string
          chat_id: string
          user_id: string | null
          role: "user" | "assistant" | "system" | "function"
          content: string
          edited: boolean
          deleted: boolean
          thread_id: string | null
          parent_id: string | null
          sent: boolean
          delivered: boolean
          read: boolean
          error_message: string | null
          retries: number
          status: "sending" | "sent" | "delivered" | "read" | "error"
          encrypted: boolean
          algorithm: string | null
          key_id: string | null
          model: string | null
          tokens: number | null
          cost: number | null
          processing_time: number | null
          confidence: number | null
          language: string | null
          sentiment: "positive" | "negative" | "neutral" | null
          topics: string[]
          created_at: string
          updated_at: string
        }
        Insert: {
          chat_id: string
          user_id?: string | null
          role: "user" | "assistant" | "system" | "function"
          content: string
          edited?: boolean
          deleted?: boolean
          thread_id?: string | null
          parent_id?: string | null
          sent?: boolean
          delivered?: boolean
          read?: boolean
          error_message?: string | null
          retries?: number
          status?: "sending" | "sent" | "delivered" | "read" | "error"
          encrypted?: boolean
          algorithm?: string | null
          key_id?: string | null
          model?: string | null
          tokens?: number | null
          cost?: number | null
          processing_time?: number | null
          confidence?: number | null
          language?: string | null
          sentiment?: "positive" | "negative" | "neutral" | null
          topics?: string[]
        }
        Update: {
          chat_id?: string
          user_id?: string | null
          role?: "user" | "assistant" | "system" | "function"
          content?: string
          edited?: boolean
          deleted?: boolean
          thread_id?: string | null
          parent_id?: string | null
          sent?: boolean
          delivered?: boolean
          read?: boolean
          error_message?: string | null
          retries?: number
          status?: "sending" | "sent" | "delivered" | "read" | "error"
          encrypted?: boolean
          algorithm?: string | null
          key_id?: string | null
          model?: string | null
          tokens?: number | null
          cost?: number | null
          processing_time?: number | null
          confidence?: number | null
          language?: string | null
          sentiment?: "positive" | "negative" | "neutral" | null
          topics?: string[]
          updated_at?: string
        }
      }
      attachments: {
        Row: {
          id: string
          message_id: string | null
          chat_id: string | null
          name: string
          type: string
          size: number
          url: string
          thumbnail_url: string | null
          content: string | null
          preview: string | null
          width: number | null
          height: number | null
          duration: number | null
          pages: number | null
          encoding: string | null
          checksum: string
          scanned: boolean
          clean: boolean
          threats: string[]
          scanned_at: string | null
          created_at: string
          uploaded_at: string
        }
        Insert: {
          message_id?: string | null
          chat_id?: string | null
          name: string
          type: string
          size: number
          url: string
          thumbnail_url?: string | null
          content?: string | null
          preview?: string | null
          width?: number | null
          height?: number | null
          duration?: number | null
          pages?: number | null
          encoding?: string | null
          checksum: string
          scanned?: boolean
          clean?: boolean
          threats?: string[]
          scanned_at?: string | null
          uploaded_at?: string
        }
        Update: {
          message_id?: string | null
          chat_id?: string | null
          name?: string
          type?: string
          size?: number
          url?: string
          thumbnail_url?: string | null
          content?: string | null
          preview?: string | null
          width?: number | null
          height?: number | null
          duration?: number | null
          pages?: number | null
          encoding?: string | null
          checksum?: string
          scanned?: boolean
          clean?: boolean
          threats?: string[]
          scanned_at?: string | null
          uploaded_at?: string
        }
      }
      reactions: {
        Row: {
          id: string
          message_id: string
          user_id: string
          type: string
          emoji: string
          created_at: string
        }
        Insert: {
          message_id: string
          user_id: string
          type: string
          emoji: string
        }
        Update: {
          message_id?: string
          user_id?: string
          type?: string
          emoji?: string
        }
      }
      mentions: {
        Row: {
          id: string
          message_id: string
          user_id: string
          start_pos: number
          end_pos: number
          type: "user" | "channel" | "file" | "url"
          created_at: string
        }
        Insert: {
          message_id: string
          user_id: string
          start_pos: number
          end_pos: number
          type: "user" | "channel" | "file" | "url"
        }
        Update: {
          message_id?: string
          user_id?: string
          start_pos?: number
          end_pos?: number
          type?: "user" | "channel" | "file" | "url"
        }
      }
      chat_versions: {
        Row: {
          id: string
          chat_id: string
          version: number
          title: string
          description: string | null
          changes: any
          parent_version: string | null
          branch: string
          tags: string[]
          created_by: string
          created_at: string
        }
        Insert: {
          chat_id: string
          version: number
          title: string
          description?: string | null
          changes?: any
          parent_version?: string | null
          branch?: string
          tags?: string[]
          created_by: string
        }
        Update: {
          chat_id?: string
          version?: number
          title?: string
          description?: string | null
          changes?: any
          parent_version?: string | null
          branch?: string
          tags?: string[]
          created_by?: string
        }
      }
      comments: {
        Row: {
          id: string
          message_id: string | null
          user_id: string
          content: string
          resolved: boolean
          parent_id: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          message_id?: string | null
          user_id: string
          content: string
          resolved?: boolean
          parent_id?: string | null
        }
        Update: {
          message_id?: string | null
          user_id?: string
          content?: string
          resolved?: boolean
          parent_id?: string | null
          updated_at?: string
        }
      }
      context_files: {
        Row: {
          id: string
          chat_id: string
          name: string
          type: string
          size: number
          path: string
          content: string | null
          added_at: string
        }
        Insert: {
          chat_id: string
          name: string
          type: string
          size: number
          path: string
          content?: string | null
        }
        Update: {
          chat_id?: string
          name?: string
          type?: string
          size?: number
          path?: string
          content?: string | null
        }
      }
      share_settings: {
        Row: {
          id: string
          chat_id: string
          is_public: boolean
          allow_comments: boolean
          allow_editing: boolean
          expires_at: string | null
          password_hash: string | null
          domain_restriction: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          chat_id: string
          is_public?: boolean
          allow_comments?: boolean
          allow_editing?: boolean
          expires_at?: string | null
          password_hash?: string | null
          domain_restriction?: string | null
        }
        Update: {
          chat_id?: string
          is_public?: boolean
          allow_comments?: boolean
          allow_editing?: boolean
          expires_at?: string | null
          password_hash?: string | null
          domain_restriction?: string | null
          updated_at?: string
        }
      }
      analytics: {
        Row: {
          id: string
          user_id: string | null
          session_id: string | null
          event: string
          properties: any
          timestamp: string
        }
        Insert: {
          user_id?: string | null
          session_id?: string | null
          event: string
          properties?: any
          timestamp?: string
        }
        Update: {
          user_id?: string | null
          session_id?: string | null
          event?: string
          properties?: any
          timestamp?: string
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}
