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

-- Saved POIs RLS Policies (join with chat_sessions to check ownership)
CREATE POLICY "Users can only see their own saved POIs" ON saved_pois
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM chat_sessions 
            WHERE chat_sessions.session_id = saved_pois.session_id 
            AND chat_sessions.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can only insert their own saved POIs" ON saved_pois
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM chat_sessions 
            WHERE chat_sessions.session_id = saved_pois.session_id 
            AND chat_sessions.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can only delete their own saved POIs" ON saved_pois
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM chat_sessions 
            WHERE chat_sessions.session_id = saved_pois.session_id 
            AND chat_sessions.user_id = auth.uid()
        )
    );

-- Sessions RLS Policies (compatibility table)
CREATE POLICY "Users can only see their own sessions" ON sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can only insert their own sessions" ON sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can only update their own sessions" ON sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can only delete their own sessions" ON sessions
    FOR DELETE USING (auth.uid() = user_id);

-- Update default user_id to use auth.uid() instead of hardcoded UUID
ALTER TABLE chat_sessions ALTER COLUMN user_id SET DEFAULT auth.uid();
ALTER TABLE messages ALTER COLUMN user_id SET DEFAULT auth.uid();
ALTER TABLE sessions ALTER COLUMN user_id SET DEFAULT auth.uid();
