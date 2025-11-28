-- Migration: Add room_code column to matches table
-- Date: 2024
-- Description: Add room_code column for PvP match joining by code

-- Add room_code column
ALTER TABLE matches 
ADD COLUMN IF NOT EXISTS room_code VARCHAR(6);

-- Create unique index on room_code (only for active matches without white_player)
-- Note: We'll handle uniqueness in application logic for active matches
CREATE INDEX IF NOT EXISTS idx_matches_room_code ON matches(room_code) 
WHERE room_code IS NOT NULL;

-- Add comment
COMMENT ON COLUMN matches.room_code IS 'Mã bàn 6 ký tự để tham gia PvP match';

