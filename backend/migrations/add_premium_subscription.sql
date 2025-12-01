-- ============================================================
-- Migration: Add premium_subscriptions table
-- ============================================================
-- Script SQL để thêm bảng premium_subscriptions vào database hiện có
-- 
-- Usage:
--   psql -U postgres -d gogame -f migrations/add_premium_subscription.sql
-- ============================================================

-- ============================================================
-- 1. TẠO BẢNG PREMIUM_SUBSCRIPTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS premium_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE NOT NULL,
    plan VARCHAR(32) NOT NULL,
    status VARCHAR(32) DEFAULT 'active' NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    -- Foreign key
    CONSTRAINT fk_premium_subscriptions_user FOREIGN KEY (user_id) 
        REFERENCES users(id) ON DELETE CASCADE,
    -- Check constraints
    CONSTRAINT chk_premium_subscriptions_plan CHECK (plan IN ('monthly', 'yearly')),
    CONSTRAINT chk_premium_subscriptions_status CHECK (status IN ('active', 'expired', 'cancelled'))
);

-- ============================================================
-- 2. TẠO INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_user_id ON premium_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_status ON premium_subscriptions(status);
CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_expires_at ON premium_subscriptions(expires_at);

-- ============================================================
-- 3. THÊM COMMENTS
-- ============================================================

COMMENT ON TABLE premium_subscriptions IS 'Bảng premium subscriptions của users';
COMMENT ON COLUMN premium_subscriptions.user_id IS 'ID của user (unique - mỗi user chỉ có 1 subscription)';
COMMENT ON COLUMN premium_subscriptions.plan IS 'Gói subscription (monthly hoặc yearly)';
COMMENT ON COLUMN premium_subscriptions.status IS 'Trạng thái (active, expired, cancelled)';
COMMENT ON COLUMN premium_subscriptions.started_at IS 'Thời gian bắt đầu subscription';
COMMENT ON COLUMN premium_subscriptions.expires_at IS 'Thời gian hết hạn subscription';
COMMENT ON COLUMN premium_subscriptions.cancelled_at IS 'Thời gian hủy subscription (nếu có)';

-- ============================================================
-- 4. VERIFY
-- ============================================================

\echo '✅ Bảng premium_subscriptions đã được tạo thành công!'
\echo ''
\echo '📊 Thông tin bảng:'
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'premium_subscriptions'
ORDER BY ordinal_position;

\echo ''
\echo '📈 Indexes:'
SELECT 
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'premium_subscriptions';

\echo ''
\echo '✅ Migration hoàn tất!'

