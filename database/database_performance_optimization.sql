-- Database Performance Optimization Script for Roameo
-- Run this in your Supabase SQL editor to optimize query performance

-- Additional indexes for better performance
-- These complement the existing basic indexes in database_schema.sql

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_messages_session_role ON messages(session_id, role);
CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_saved_pois_session_poi ON saved_pois(session_id, poi_id);

-- JSONB indexes for trip data queries
CREATE INDEX IF NOT EXISTS idx_chat_sessions_trip_destination ON chat_sessions USING gin ((trip->>'destination'));
CREATE INDEX IF NOT EXISTS idx_chat_sessions_trip_origin ON chat_sessions USING gin ((trip->>'origin'));
CREATE INDEX IF NOT EXISTS idx_sessions_trip_destination ON sessions USING gin ((trip->>'destination'));
CREATE INDEX IF NOT EXISTS idx_sessions_trip_origin ON sessions USING gin ((trip->>'origin'));

-- Partial indexes for active sessions (sessions with recent activity)
CREATE INDEX IF NOT EXISTS idx_chat_sessions_recent_active 
ON chat_sessions(updated_at DESC) 
WHERE updated_at > NOW() - INTERVAL '7 days';

CREATE INDEX IF NOT EXISTS idx_sessions_recent_active 
ON sessions(updated_at DESC) 
WHERE updated_at > NOW() - INTERVAL '7 days';

-- Index for message ordering within sessions (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_messages_session_time_desc ON messages(session_id, created_at DESC);

-- Covering index for session lookup with all commonly accessed fields
CREATE INDEX IF NOT EXISTS idx_chat_sessions_lookup 
ON chat_sessions(session_id) 
INCLUDE (invite_id, trip, created_at, updated_at);

-- Performance optimization for bulk operations
-- Create function to update multiple trip fields efficiently
CREATE OR REPLACE FUNCTION update_trip_fields(
  p_session_id TEXT,
  p_destination TEXT DEFAULT NULL,
  p_origin TEXT DEFAULT NULL,
  p_days INTEGER DEFAULT NULL,
  p_travelers INTEGER DEFAULT NULL,
  p_budget TEXT DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
  current_trip JSONB;
  updated_trip JSONB;
BEGIN
  -- Get current trip data
  SELECT trip INTO current_trip 
  FROM chat_sessions 
  WHERE session_id = p_session_id;
  
  -- Initialize if null
  IF current_trip IS NULL THEN
    current_trip := '{}'::jsonb;
  END IF;
  
  -- Build updated trip object
  updated_trip := current_trip;
  
  IF p_destination IS NOT NULL THEN
    updated_trip := jsonb_set(updated_trip, '{destination}', to_jsonb(p_destination));
  END IF;
  
  IF p_origin IS NOT NULL THEN
    updated_trip := jsonb_set(updated_trip, '{origin}', to_jsonb(p_origin));
  END IF;
  
  IF p_days IS NOT NULL THEN
    updated_trip := jsonb_set(updated_trip, '{days}', to_jsonb(p_days));
  END IF;
  
  IF p_travelers IS NOT NULL THEN
    updated_trip := jsonb_set(updated_trip, '{travelers}', to_jsonb(p_travelers));
  END IF;
  
  IF p_budget IS NOT NULL THEN
    updated_trip := jsonb_set(updated_trip, '{budget}', to_jsonb(p_budget));
  END IF;
  
  -- Update the session
  UPDATE chat_sessions 
  SET trip = updated_trip, updated_at = NOW()
  WHERE session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;

-- Function to efficiently get session with recent messages
CREATE OR REPLACE FUNCTION get_session_with_recent_messages(
  p_session_id TEXT,
  p_message_limit INTEGER DEFAULT 50
)
RETURNS TABLE(
  session_id TEXT,
  invite_id TEXT,
  trip JSONB,
  created_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ,
  messages JSONB
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    cs.session_id,
    cs.invite_id,
    cs.trip,
    cs.created_at,
    cs.updated_at,
    COALESCE(
      (SELECT jsonb_agg(
        jsonb_build_object(
          'id', m.id,
          'role', m.role,
          'content', m.content,
          'createdAt', m.created_at
        ) ORDER BY m.created_at DESC
      )
      FROM (
        SELECT id, role, content, created_at
        FROM messages 
        WHERE messages.session_id = p_session_id
        ORDER BY created_at DESC
        LIMIT p_message_limit
      ) m),
      '[]'::jsonb
    ) as messages
  FROM chat_sessions cs
  WHERE cs.session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old inactive sessions (for maintenance)
CREATE OR REPLACE FUNCTION cleanup_inactive_sessions(
  p_days_threshold INTEGER DEFAULT 30
)
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  -- Delete sessions that haven't been updated in the specified days
  WITH deleted_sessions AS (
    DELETE FROM chat_sessions 
    WHERE updated_at < NOW() - (p_days_threshold || ' days')::INTERVAL
    RETURNING session_id
  ),
  deleted_messages AS (
    DELETE FROM messages 
    WHERE session_id IN (SELECT session_id FROM deleted_sessions)
    RETURNING 1
  ),
  deleted_pois AS (
    DELETE FROM saved_pois 
    WHERE session_id IN (SELECT session_id FROM deleted_sessions)
    RETURNING 1
  )
  SELECT COUNT(*) INTO deleted_count FROM deleted_sessions;
  
  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Add statistics collection for query optimization
-- This helps PostgreSQL's query planner make better decisions
ANALYZE chat_sessions;
ANALYZE messages;
ANALYZE saved_pois;
ANALYZE sessions;

-- Enable auto-vacuum for better performance
-- (This is usually enabled by default in Supabase, but ensuring it's set)
ALTER TABLE chat_sessions SET (autovacuum_enabled = true);
ALTER TABLE messages SET (autovacuum_enabled = true);
ALTER TABLE saved_pois SET (autovacuum_enabled = true);
ALTER TABLE sessions SET (autovacuum_enabled = true);

-- Set table statistics targets for better query planning
ALTER TABLE chat_sessions ALTER COLUMN trip SET STATISTICS 1000;
ALTER TABLE chat_sessions ALTER COLUMN session_id SET STATISTICS 1000;
ALTER TABLE messages ALTER COLUMN session_id SET STATISTICS 1000;
ALTER TABLE messages ALTER COLUMN content SET STATISTICS 100;

-- Create a view for active sessions with message counts (for dashboard/analytics)
CREATE OR REPLACE VIEW active_sessions_summary AS
SELECT 
  cs.session_id,
  cs.invite_id,
  cs.trip->>'destination' as destination,
  cs.trip->>'origin' as origin,
  cs.created_at,
  cs.updated_at,
  COUNT(m.id) as message_count,
  COUNT(sp.poi_id) as saved_poi_count,
  MAX(m.created_at) as last_message_at
FROM chat_sessions cs
LEFT JOIN messages m ON m.session_id = cs.session_id
LEFT JOIN saved_pois sp ON sp.session_id = cs.session_id
WHERE cs.updated_at > NOW() - INTERVAL '7 days'
GROUP BY cs.session_id, cs.invite_id, cs.trip, cs.created_at, cs.updated_at;

-- Grant necessary permissions for the view
-- (Adjust according to your RLS policies)
-- GRANT SELECT ON active_sessions_summary TO authenticated;

COMMENT ON INDEX idx_messages_session_role IS 'Optimizes queries filtering messages by session and role';
COMMENT ON INDEX idx_messages_session_created IS 'Optimizes message ordering within sessions';
COMMENT ON INDEX idx_chat_sessions_trip_destination IS 'Enables fast searches by destination in trip data';
COMMENT ON INDEX idx_chat_sessions_recent_active IS 'Optimizes queries for recently active sessions';
COMMENT ON FUNCTION update_trip_fields IS 'Efficiently updates multiple trip fields in a single operation';
COMMENT ON FUNCTION get_session_with_recent_messages IS 'Retrieves session data with recent messages in a single query';
COMMENT ON VIEW active_sessions_summary IS 'Provides summary statistics for active sessions';