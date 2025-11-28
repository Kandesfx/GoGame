-- Migration: Add time control columns to matches table
-- Run this SQL script if Alembic is not available

ALTER TABLE matches
ADD COLUMN IF NOT EXISTS time_control_minutes INTEGER,
ADD COLUMN IF NOT EXISTS black_time_remaining_seconds INTEGER,
ADD COLUMN IF NOT EXISTS white_time_remaining_seconds INTEGER,
ADD COLUMN IF NOT EXISTS last_move_at TIMESTAMP WITH TIME ZONE;

-- Add comments for documentation
COMMENT ON COLUMN matches.time_control_minutes IS 'Thời gian tổng cho mỗi người chơi (phút)';
COMMENT ON COLUMN matches.black_time_remaining_seconds IS 'Thời gian còn lại của Black (giây)';
COMMENT ON COLUMN matches.white_time_remaining_seconds IS 'Thời gian còn lại của White (giây)';
COMMENT ON COLUMN matches.last_move_at IS 'Thời điểm nước đi cuối cùng';

