-- Script để chuyển đổi các cột UUID từ VARCHAR sang UUID
-- Chạy script này nếu gặp lỗi "operator does not exist: character varying = uuid"

-- Lưu ý: Script này sẽ chuyển đổi kiểu dữ liệu của các cột UUID
-- Nếu có foreign key constraints, cần xóa và tạo lại

DO $$ 
BEGIN
    -- 1. Chuyển đổi bảng users (phải làm trước vì các bảng khác reference nó)
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' 
        AND column_name = 'id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting users.id from VARCHAR to UUID...';
        
        -- Xóa foreign key constraints từ các bảng khác trước
        -- (sẽ tạo lại sau)
        
        -- Chuyển đổi cột id
        ALTER TABLE users ALTER COLUMN id TYPE UUID USING id::UUID;
        RAISE NOTICE '✅ users.id converted to UUID';
    END IF;
    
    -- 2. Chuyển đổi bảng matches
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'matches' 
        AND column_name = 'id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting matches.id from VARCHAR to UUID...';
        ALTER TABLE matches ALTER COLUMN id TYPE UUID USING id::UUID;
        RAISE NOTICE '✅ matches.id converted to UUID';
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'matches' 
        AND column_name = 'black_player_id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting matches.black_player_id from VARCHAR to UUID...';
        ALTER TABLE matches ALTER COLUMN black_player_id TYPE UUID USING black_player_id::UUID;
        RAISE NOTICE '✅ matches.black_player_id converted to UUID';
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'matches' 
        AND column_name = 'white_player_id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting matches.white_player_id from VARCHAR to UUID...';
        ALTER TABLE matches ALTER COLUMN white_player_id TYPE UUID USING white_player_id::UUID;
        RAISE NOTICE '✅ matches.white_player_id converted to UUID';
    END IF;
    
    -- 3. Chuyển đổi bảng refresh_tokens
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'refresh_tokens' 
        AND column_name = 'id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting refresh_tokens.id from VARCHAR to UUID...';
        ALTER TABLE refresh_tokens ALTER COLUMN id TYPE UUID USING id::UUID;
        RAISE NOTICE '✅ refresh_tokens.id converted to UUID';
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'refresh_tokens' 
        AND column_name = 'user_id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting refresh_tokens.user_id from VARCHAR to UUID...';
        ALTER TABLE refresh_tokens ALTER COLUMN user_id TYPE UUID USING user_id::UUID;
        RAISE NOTICE '✅ refresh_tokens.user_id converted to UUID';
    END IF;
    
    -- 4. Chuyển đổi bảng coin_transactions
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coin_transactions' 
        AND column_name = 'id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting coin_transactions.id from VARCHAR to UUID...';
        ALTER TABLE coin_transactions ALTER COLUMN id TYPE UUID USING id::UUID;
        RAISE NOTICE '✅ coin_transactions.id converted to UUID';
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coin_transactions' 
        AND column_name = 'user_id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting coin_transactions.user_id from VARCHAR to UUID...';
        ALTER TABLE coin_transactions ALTER COLUMN user_id TYPE UUID USING user_id::UUID;
        RAISE NOTICE '✅ coin_transactions.user_id converted to UUID';
    END IF;
    
    -- 5. Chuyển đổi bảng premium_requests
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'premium_requests' 
        AND column_name = 'id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting premium_requests.id from VARCHAR to UUID...';
        ALTER TABLE premium_requests ALTER COLUMN id TYPE UUID USING id::UUID;
        RAISE NOTICE '✅ premium_requests.id converted to UUID';
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'premium_requests' 
        AND column_name = 'user_id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting premium_requests.user_id from VARCHAR to UUID...';
        ALTER TABLE premium_requests ALTER COLUMN user_id TYPE UUID USING user_id::UUID;
        RAISE NOTICE '✅ premium_requests.user_id converted to UUID';
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'premium_requests' 
        AND column_name = 'match_id' 
        AND data_type = 'character varying'
    ) THEN
        RAISE NOTICE 'Converting premium_requests.match_id from VARCHAR to UUID...';
        ALTER TABLE premium_requests ALTER COLUMN match_id TYPE UUID USING match_id::UUID;
        RAISE NOTICE '✅ premium_requests.match_id converted to UUID';
    END IF;
    
    RAISE NOTICE '✅ All UUID conversions completed!';
END $$;

