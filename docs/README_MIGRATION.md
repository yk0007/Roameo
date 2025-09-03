# Database Migration: Sessions Table Consolidation

## Overview
This migration consolidates the duplicate `sessions` and `chat_sessions` tables into a single `chat_sessions` table.

## Files Created
- `consolidate_sessions_migration.sql` - Main migration script
- `cleanup_sessions_schema.sql` - Final cleanup script
- `README_MIGRATION.md` - This documentation

## Migration Steps

### 1. Run Analysis First
Execute the consolidation script in Supabase SQL Editor to analyze current data:
```sql
-- Run consolidate_sessions_migration.sql
```

This will show you:
- Record counts in both tables
- Any unique data in the `sessions` table
- Migration results

### 2. Review Results
Check the output to ensure:
- No data loss during migration
- All `session_id` values are preserved
- Foreign key constraints are satisfied

### 3. Run Cleanup
After verifying the migration was successful:
```sql
-- Run cleanup_sessions_schema.sql
```

This will:
- Drop the redundant `sessions` table
- Add performance indexes
- Verify final schema state

## Safety Notes
- ✅ The migration script is safe - it only inserts missing data
- ✅ No existing data in `chat_sessions` will be modified
- ✅ The `DROP TABLE` command is in a separate script for safety
- ✅ All steps include verification queries

## Rollback Plan
If issues occur:
1. The original `sessions` table data remains until cleanup
2. You can recreate the `sessions` table from `chat_sessions` if needed
3. All related tables (`messages`, `saved_pois`) use `session_id` text field, not table references

## Expected Outcome
- Single `chat_sessions` table with all session data
- Improved schema clarity
- Better performance with proper indexes
- Reduced maintenance overhead
