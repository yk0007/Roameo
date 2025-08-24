-- Add user_id columns to all tables first
-- This must be run BEFORE the RLS policies

-- Add user_id column to chat_sessions if it doesn't exist
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS user_id UUID;

-- Add user_id column to messages if it doesn't exist  
ALTER TABLE messages ADD COLUMN IF NOT EXISTS user_id UUID;

-- Add user_id column to saved_pois if it doesn't exist
ALTER TABLE saved_pois ADD COLUMN IF NOT EXISTS user_id UUID;

-- Add user_id column to sessions if it doesn't exist
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id UUID;

-- Set default values for user_id columns
ALTER TABLE chat_sessions ALTER COLUMN user_id SET DEFAULT auth.uid();
ALTER TABLE messages ALTER COLUMN user_id SET DEFAULT auth.uid();
ALTER TABLE saved_pois ALTER COLUMN user_id SET DEFAULT auth.uid();
ALTER TABLE sessions ALTER COLUMN user_id SET DEFAULT auth.uid();

-- Update existing records to have a default user_id (temporary for migration)
-- You may want to update these to actual user IDs if you have that data
UPDATE chat_sessions SET user_id = '00000000-0000-0000-0000-000000000000'::uuid WHERE user_id IS NULL;
UPDATE messages SET user_id = '00000000-0000-0000-0000-000000000000'::uuid WHERE user_id IS NULL;
UPDATE saved_pois SET user_id = '00000000-0000-0000-0000-000000000000'::uuid WHERE user_id IS NULL;
UPDATE sessions SET user_id = '00000000-0000-0000-0000-000000000000'::uuid WHERE user_id IS NULL;
