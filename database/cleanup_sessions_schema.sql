-- Final cleanup script to run AFTER consolidate_sessions_migration.sql
-- This removes the duplicate sessions table and ensures clean schema

-- Step 1: Drop the redundant sessions table
DROP TABLE IF EXISTS sessions;

-- Step 2: Add any missing indexes for performance on chat_sessions
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at);

-- Step 3: Update RLS policies to ensure they reference the correct table
-- (These should already be correct based on your previous fixes)

-- Step 4: Verify final schema state
SELECT 
  schemaname,
  tablename,
  tableowner
FROM pg_tables 
WHERE tablename IN ('sessions', 'chat_sessions', 'messages', 'saved_pois')
ORDER BY tablename;
