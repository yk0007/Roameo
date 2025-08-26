-- Migration script to consolidate duplicate sessions tables
-- This script safely merges 'sessions' table data into 'chat_sessions' and removes the duplicate

-- Step 1: Check if there's any data in the 'sessions' table that's not in 'chat_sessions'
-- Run this first to see what data exists
SELECT 
  'sessions' as source_table,
  COUNT(*) as record_count,
  MIN(created_at) as earliest_record,
  MAX(created_at) as latest_record
FROM sessions
UNION ALL
SELECT 
  'chat_sessions' as source_table,
  COUNT(*) as record_count,
  MIN(created_at) as earliest_record,
  MAX(created_at) as latest_record
FROM chat_sessions;

-- Step 2: Find any sessions records that don't exist in chat_sessions
-- (based on session_id which should be unique across both tables)
SELECT s.* 
FROM sessions s
LEFT JOIN chat_sessions cs ON s.session_id = cs.session_id
WHERE cs.session_id IS NULL;

-- Step 3: Migrate any unique data from sessions to chat_sessions
-- Only insert records that don't already exist in chat_sessions
INSERT INTO chat_sessions (session_id, user_id, invite_id, trip, created_at, updated_at)
SELECT s.session_id, s.user_id, s.invite_id, s.trip, s.created_at, s.updated_at
FROM sessions s
LEFT JOIN chat_sessions cs ON s.session_id = cs.session_id
WHERE cs.session_id IS NULL;

-- Step 4: Verify the migration was successful
-- Check that all session_ids from sessions table now exist in chat_sessions
SELECT 
  COUNT(*) as sessions_count,
  COUNT(cs.session_id) as found_in_chat_sessions
FROM sessions s
LEFT JOIN chat_sessions cs ON s.session_id = cs.session_id;

-- Step 5: Check for any foreign key references to the sessions table
-- (This should be empty based on the schema, but let's verify)
SELECT 
  tc.table_name, 
  kcu.column_name,
  ccu.table_name AS foreign_table_name,
  ccu.column_name AS foreign_column_name 
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
  ON tc.constraint_name = kcu.constraint_name
  AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
  ON ccu.constraint_name = tc.constraint_name
  AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' 
  AND ccu.table_name = 'sessions';

-- Step 6: Drop the redundant sessions table
-- IMPORTANT: Only run this after verifying steps 1-5 are successful
-- DROP TABLE sessions;

-- Step 7: Verify final state
-- SELECT COUNT(*) as total_chat_sessions FROM chat_sessions;
