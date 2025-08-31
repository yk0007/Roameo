-- Step 1: First run add_user_columns.sql to add user_id columns
-- Step 2: Then run this file to create RLS policies

-- Enable RLS on all tables
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_pois ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can only see their own chat sessions" ON chat_sessions;
DROP POLICY IF EXISTS "Users can only insert their own chat sessions" ON chat_sessions;
DROP POLICY IF EXISTS "Users can only update their own chat sessions" ON chat_sessions;
DROP POLICY IF EXISTS "Users can only delete their own chat sessions" ON chat_sessions;

DROP POLICY IF EXISTS "Users can only see their own messages" ON messages;
DROP POLICY IF EXISTS "Users can only insert their own messages" ON messages;
DROP POLICY IF EXISTS "Users can only update their own messages" ON messages;
DROP POLICY IF EXISTS "Users can only delete their own messages" ON messages;

DROP POLICY IF EXISTS "Users can only see their own saved POIs" ON saved_pois;
DROP POLICY IF EXISTS "Users can only insert their own saved POIs" ON saved_pois;
DROP POLICY IF EXISTS "Users can only delete their own saved POIs" ON saved_pois;

DROP POLICY IF EXISTS "Users can only see their own sessions" ON sessions;
DROP POLICY IF EXISTS "Users can only insert their own sessions" ON sessions;
DROP POLICY IF EXISTS "Users can only update their own sessions" ON sessions;
DROP POLICY IF EXISTS "Users can only delete their own sessions" ON sessions;

-- Chat Sessions RLS Policies
CREATE POLICY "Users can only see their own chat sessions" ON chat_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can only insert their own chat sessions" ON chat_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can only update their own chat sessions" ON chat_sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can only delete their own chat sessions" ON chat_sessions
    FOR DELETE USING (auth.uid() = user_id);

-- Messages RLS Policies
CREATE POLICY "Users can only see their own messages" ON messages
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can only insert their own messages" ON messages
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can only update their own messages" ON messages
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can only delete their own messages" ON messages
    FOR DELETE USING (auth.uid() = user_id);

-- Saved POIs RLS Policies
CREATE POLICY "Users can only see their own saved POIs" ON saved_pois
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can only insert their own saved POIs" ON saved_pois
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can only delete their own saved POIs" ON saved_pois
    FOR DELETE USING (auth.uid() = user_id);

-- Sessions RLS Policies
CREATE POLICY "Users can only see their own sessions" ON sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can only insert their own sessions" ON sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can only update their own sessions" ON sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can only delete their own sessions" ON sessions
    FOR DELETE USING (auth.uid() = user_id);
