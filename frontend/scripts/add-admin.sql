-- Admin User Creation Script for Archyx AI
-- This script creates an admin user with specified credentials
-- Email: admin@example.com
-- Password: admin_example_password

-- =====================================================
-- CREATE ADMIN USER (AUTH + PROFILE)
-- =====================================================

DO $$
DECLARE
  admin_user_id UUID;
  hashed_password TEXT;
BEGIN
  -- Generate a consistent UUID for the admin user
  admin_user_id := '00000000-0000-0000-0000-000000000001'::UUID;
  
  -- Hash the password using crypt (bcrypt)
  hashed_password := crypt('admin_example_password', gen_salt('bf'));
  
  -- Insert into auth.users with proper metadata to work with trigger
  INSERT INTO auth.users (
    id,
    instance_id,
    aud,
    role,
    email,
    encrypted_password,
    email_confirmed_at,
    created_at,
    updated_at,
    confirmation_token,
    email_change,
    email_change_token_new,
    recovery_token,
    raw_user_meta_data
  ) VALUES (
    admin_user_id,
    '00000000-0000-0000-0000-000000000000'::UUID,
    'authenticated',
    'authenticated',
    'admin@example.com',
    hashed_password,
    NOW(),
    NOW(),
    NOW(),
    '',
    '',
    '',
    '',
    jsonb_build_object(
      'full_name', 'System Administrator',
      'is_admin', true,
      'status', 'active',
      'permissions', jsonb_build_array('admin_access', 'ai_chat', 'file_organization', 'version_history', 'context_management', 'tools_access')
    )
  ) ON CONFLICT (id) DO UPDATE SET
    email = EXCLUDED.email,
    encrypted_password = EXCLUDED.encrypted_password,
    raw_user_meta_data = EXCLUDED.raw_user_meta_data,
    updated_at = NOW();  
  -- Insert into auth.identities (check if it already exists first)
  INSERT INTO auth.identities (
    provider_id,
    user_id,
    identity_data,
    provider,
    created_at,
    updated_at
  ) 
  SELECT 
    admin_user_id::text,
    admin_user_id,
    jsonb_build_object('sub', admin_user_id::text, 'email', 'admin@example.com'),
    'email',
    NOW(),
    NOW()
  WHERE NOT EXISTS (
    SELECT 1 FROM auth.identities 
    WHERE user_id = admin_user_id AND provider = 'email'
  );
  
  -- Update existing identity if it exists
  UPDATE auth.identities SET
    identity_data = jsonb_build_object('sub', admin_user_id::text, 'email', 'admin@example.com'),
    updated_at = NOW()
  WHERE user_id = admin_user_id AND provider = 'email';
  
  -- Update the profile created by trigger with admin-specific settings
  UPDATE public.profiles SET
    full_name = 'System Administrator',
    is_admin = TRUE,
    status = 'active',
    permissions = ARRAY['admin_access', 'ai_chat', 'file_organization', 'version_history', 'context_management', 'tools_access']::TEXT[],
    subscription_plan = 'enterprise',
    subscription_features = ARRAY['basic_chat', 'file_upload', 'advanced_models', 'custom_models', 'priority_support']::TEXT[],
    billing_currency = 'USD',
    messages_per_day = 10000,
    tokens_per_month = 10000000,
    file_uploads = 1000,
    custom_models = 50,
    last_active = NOW(),
    updated_at = NOW()
  WHERE id = admin_user_id;  
  -- Update user settings (created by trigger) with admin preferences
  UPDATE user_settings SET
    default_model = 'gpt-4',
    communication_style = 'professional',
    response_length = 'detailed',
    expertise_level = 'expert',
    updated_at = NOW()
  WHERE user_id = admin_user_id;
  
  -- Create welcome chat for admin (only if it doesn't exist)
  IF NOT EXISTS (SELECT 1 FROM chats WHERE user_id = admin_user_id AND title = 'Admin Welcome - Archyx AI') THEN
    INSERT INTO chats (
      user_id,
      title,
      description,
      category,
      model,
      system_prompt
    ) VALUES (
      admin_user_id,
      'Admin Welcome - Archyx AI',
      'Administrator welcome chat with system overview',
      'admin',
      'gpt-4',
      'You are an AI assistant for the Archyx AI system administrator. Provide detailed system information and help with administrative tasks.'
    );
  END IF;
  
  RAISE NOTICE 'Admin user created/updated successfully with ID: %', admin_user_id;
  RAISE NOTICE 'Admin credentials: admin@example.com / admin_example_password';
END $$;

-- =====================================================
-- VERIFICATION QUERIES
-- =====================================================

-- Verify the admin user was created successfully
SELECT 
  p.id,
  p.email,
  p.full_name,
  p.is_admin,
  p.status,
  p.permissions,
  p.subscription_plan,
  p.created_at
FROM profiles p
WHERE p.email = 'admin@example.com';

-- Check user settings were created
SELECT 
  us.user_id,
  us.default_model,
  us.communication_style,
  us.response_length,
  us.expertise_level
FROM user_settings us
JOIN profiles p ON p.id = us.user_id
WHERE p.email = 'admin@example.com';

-- Check welcome chat was created
SELECT 
  c.id,
  c.title,
  c.description,
  c.category,
  c.model,
  c.created_at
FROM chats c
JOIN profiles p ON p.id = c.user_id
WHERE p.email = 'admin@example.com';

-- =====================================================
-- SIMPLIFIED SETUP INSTRUCTIONS
-- =====================================================

/*
SETUP IS NOW AUTOMATED:

Just run this SQL script in your Supabase SQL Editor - no manual steps needed!

This script will:
1. Create the auth user directly in the database
2. Set up the admin profile with full permissions
3. Create user settings optimized for admin use
4. Add a welcome chat for the administrator

ADMIN USER CREDENTIALS:
Email: admin@example.com
Password: admin_example_password

ADMIN PERMISSIONS:
- admin_access: Full administrative access
- ai_chat: Can use AI chat functionality
- file_organization: Can organize files and folders
- version_history: Can manage version history
- context_management: Can manage context
- tools_access: Can access additional tools

SUBSCRIPTION: Enterprise plan with maximum limits
- Messages per day: 10,000
- Tokens per month: 10,000,000
- File uploads: 1,000
- Custom models: 50

SECURITY NOTE: 
Please change the default password after first login for security.
The admin user has full access to the system including user management.

The script is idempotent - you can run it multiple times safely.
*/
