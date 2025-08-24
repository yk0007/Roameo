-- Update existing chat sessions to have proper user_id values
-- This script fixes sessions that have placeholder user_id values

-- First, let's see what we have
-- SELECT session_id, user_id, created_at FROM chat_sessions WHERE user_id = '00000000-0000-0000-0000-000000000000';

-- For now, we'll delete sessions with placeholder user_id since they can't be properly attributed
-- You can run this after creating new sessions with proper authentication
DELETE FROM chat_sessions WHERE user_id = '00000000-0000-0000-0000-000000000000';
DELETE FROM messages WHERE user_id = '00000000-0000-0000-0000-000000000000';
DELETE FROM saved_pois WHERE user_id = '00000000-0000-0000-0000-000000000000';
DELETE FROM sessions WHERE user_id = '00000000-0000-0000-0000-000000000000';

-- Alternative: If you want to keep the data and assign it to a specific user, 
-- replace 'YOUR_ACTUAL_USER_ID' with the real user ID from auth.users table:
-- UPDATE chat_sessions SET user_id = 'YOUR_ACTUAL_USER_ID' WHERE user_id = '00000000-0000-0000-0000-000000000000';
-- UPDATE messages SET user_id = 'YOUR_ACTUAL_USER_ID' WHERE user_id = '00000000-0000-0000-0000-000000000000';
-- UPDATE saved_pois SET user_id = 'YOUR_ACTUAL_USER_ID' WHERE user_id = '00000000-0000-0000-0000-000000000000';
-- UPDATE sessions SET user_id = 'YOUR_ACTUAL_USER_ID' WHERE user_id = '00000000-0000-0000-0000-000000000000';
