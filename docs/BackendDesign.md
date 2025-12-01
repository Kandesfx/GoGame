# Backend Architecture Design — GoGame

This document describes the backend/API layer that complements the existing C++ AI engines and Python ML modules. The design follows the requirements in `SystemSpec.md` (Sections 1.8, 3.5 and monetisation strategy) and focuses on a FastAPI-based service with PostgreSQL + MongoDB.

**Note:** Database schema trong tài liệu này đã được cập nhật để phù hợp với implementation thực tế. Xem `backend/scripts/database_schema.sql` để biết schema đầy đủ và chi tiết.

---

## 1. Goals & Scope

**Primary responsibilities**
1. Authentication & user profile management.
2. Match orchestration (PvP, PvAI, premium analysis requests).
3. Monetisation (coin system, transactions, premium feature access).
4. Storage of structured data (PostgreSQL), semi-structured game data (MongoDB) and model artefacts (S3/MinIO).
5. Provide REST/WS APIs for frontend clients (desktop UI) and future mobile/web apps.

**Out of scope (phase 1)**
- Real-time matchmaking queue (use simple challenge-accept).
- Payment gateway integration (stubbed).
- High-concurrency scaling (one-process deployment sufficient).

---

## 2. Technology Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Web framework | FastAPI (Python 3.10+) | Async, easy integration with Pydantic & dependency injection. |
| Task queue | FastAPI background tasks (phase 1), optional Celery for future heavy ML jobs. |
| ORM / DB toolkit | SQLAlchemy 2.0 + Alembic migrations. |
| Relational DB | PostgreSQL 14+, stores users, matches metadata, coin transactions. |
| Document DB | MongoDB 6+, stores SGF/game logs, premium analysis reports. |
| Object storage | MinIO/S3 for ML checkpoints & large files. |
| Authentication | JWT (Access+Refresh) using PyJWT, password hashing via Argon2. |
| Realtime | WebSocket endpoints via FastAPI for live game updates (future). |

---

## 3. Deployment Topology

```
┌────────────────────────────────────────────┐
│                 Frontend                   │
│  Electron/PyQt Client  ←→  FastAPI REST    │
└────────────────────────┬───────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │       FastAPI Application      │
         │  - app/main.py                 │
         │  - routers/                    │
         │  - services/                   │
         │  - dependency injection        │
         └───────┬───────────────┬────────┘
                 │               │
     ┌───────────▼──────┐ ┌──────▼──────────────────┐
     │ PostgreSQL       │ │ MongoDB                 │
     │ (SQLAlchemy)     │ │ (Motor/MongoEngine)     │
     └───────────┬──────┘ └──────────────┬──────────┘
                 │                       │
         ┌───────▼────────┐     ┌────────▼──────────┐
         │ Redis (optional│     │ S3/MinIO (models, │
         │ for cache)     │     │ analysis exports) │
         └────────────────┘     └───────────────────┘
```

---

## 4. Codebase Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI app factory
│   ├── config.py               # Settings (Pydantic BaseSettings)
│   ├── database.py             # SessionLocal, Mongo client
│   ├── dependencies.py         # Common DI helpers
│   ├── routers/
│   │   ├── auth.py             # /auth endpoints
│   │   ├── users.py            # /users
│   │   ├── matches.py          # /matches, /pvp, /ai
│   │   ├── premium.py          # /premium/hint, /analysis, /shop
│   │   ├── coins.py            # /coins purchase & history
│   │   └── ml.py               # /ml/models, /ml/train (future)
│   ├── services/
│   │   ├── auth_service.py     # JWT issuance, password hashing
│   │   ├── user_service.py
│   │   ├── match_service.py    # orchestrates AI engine calls
│   │   ├── premium_service.py  # wraps gogame_py AI hints
│   │   ├── coin_service.py
│   │   └── ml_service.py
│   ├── models/
│   │   ├── sql/                # SQLAlchemy ORM models
│   │   └── mongo/              # Pydantic models for Mongo docs
│   ├── schemas/                # Pydantic request/response models
│   ├── utils/                  # common helpers (SGF, hashing)
│   └── tasks/                  # background task definitions
├── migrations/                 # Alembic
├── tests/                      # API integration tests (pytest)
└── requirements.txt / poetry   # dependencies
```

---

## 5. Database Schema

### 5.1 PostgreSQL Schema

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(32) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    elo_rating INTEGER DEFAULT 1500 NOT NULL,
    coins INTEGER DEFAULT 0 NOT NULL,
    display_name VARCHAR(64),
    avatar_url VARCHAR(255),
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Refresh tokens table
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    token TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    revoked BOOLEAN DEFAULT FALSE NOT NULL
);

-- Matches table
CREATE TABLE matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    black_player_id UUID REFERENCES users (id) ON DELETE SET NULL,
    white_player_id UUID REFERENCES users (id) ON DELETE SET NULL,
    ai_level INTEGER,
    board_size INTEGER DEFAULT 9 NOT NULL,
    result VARCHAR(32),
    room_code VARCHAR(6) UNIQUE,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    finished_at TIMESTAMP WITH TIME ZONE,
    sgf_id VARCHAR(64),
    premium_analysis_id VARCHAR(64),
    -- Time control cho PvP matches
    time_control_minutes INTEGER,
    black_time_remaining_seconds INTEGER,
    white_time_remaining_seconds INTEGER,
    last_move_at TIMESTAMP WITH TIME ZONE,
    -- ELO changes (chỉ cho PvP matches)
    black_elo_change INTEGER,
    white_elo_change INTEGER,
    -- Ready status cho matchmaking (chỉ cho PvP matches)
    black_ready BOOLEAN DEFAULT FALSE NOT NULL,
    white_ready BOOLEAN DEFAULT FALSE NOT NULL
);

-- Coin transactions table
CREATE TABLE coin_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    amount INTEGER NOT NULL,
    type VARCHAR(32) NOT NULL,
    source VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    CONSTRAINT chk_coin_transactions_amount CHECK (amount != 0)
);

-- Premium requests table
CREATE TABLE premium_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    match_id UUID NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    feature VARCHAR(32) NOT NULL,
    cost INTEGER NOT NULL,
    status VARCHAR(32) DEFAULT 'pending' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Alembic version tracking
CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL PRIMARY KEY
);

-- Indexes
CREATE INDEX ix_users_username ON users(username);
CREATE INDEX ix_users_email ON users(email);
CREATE INDEX ix_users_elo_rating ON users(elo_rating);
CREATE INDEX ix_matches_room_code ON matches(room_code) WHERE room_code IS NOT NULL;
CREATE INDEX ix_matches_black_player_id ON matches(black_player_id) WHERE black_player_id IS NOT NULL;
CREATE INDEX ix_matches_white_player_id ON matches(white_player_id) WHERE white_player_id IS NOT NULL;
CREATE INDEX ix_matches_started_at ON matches(started_at);
CREATE INDEX ix_matches_finished_at ON matches(finished_at) WHERE finished_at IS NOT NULL;
CREATE INDEX ix_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX ix_refresh_tokens_token ON refresh_tokens(token);
CREATE INDEX ix_refresh_tokens_expires_at ON refresh_tokens(expires_at);
CREATE INDEX ix_coin_transactions_user_id ON coin_transactions(user_id);
CREATE INDEX ix_coin_transactions_created_at ON coin_transactions(created_at);
CREATE INDEX ix_coin_transactions_type ON coin_transactions(type);
CREATE INDEX ix_premium_requests_user_id ON premium_requests(user_id);
CREATE INDEX ix_premium_requests_match_id ON premium_requests(match_id);
CREATE INDEX ix_premium_requests_status ON premium_requests(status);
CREATE INDEX ix_premium_requests_created_at ON premium_requests(created_at);
```

**Lưu ý:**
- Tất cả ID sử dụng `UUID` type (PostgreSQL native UUID type) để tối ưu hiệu suất và tự động validate format.
- Bảng `users` có thêm các trường: `display_name`, `avatar_url`, `preferences` (JSONB).
- Bảng `matches` có thêm các trường cho matchmaking và time control: `room_code`, `time_control_minutes`, `black_time_remaining_seconds`, `white_time_remaining_seconds`, `last_move_at`, `black_elo_change`, `white_elo_change`, `black_ready`, `white_ready`.
- Bảng `refresh_tokens` sử dụng `TEXT` cho token (hỗ trợ token dài hơn) và có trường `revoked`.
- Bảng `coin_transactions` sử dụng `type` và `source` thay vì `transaction_type` và `description`.
- Bảng `premium_requests` có trường `feature` và `cost` để xác định tính năng và chi phí.
- Tất cả các bảng đều có indexes phù hợp để tối ưu hiệu suất truy vấn.
- Bảng `alembic_version` được sử dụng để tracking Alembic migrations.

**Xem thêm:** Để biết schema đầy đủ và chi tiết, xem `backend/scripts/database_schema.sql`.

### 5.2 MongoDB Collections

```javascript
// games collection
{
  _id: ObjectId(),
  match_id: UUID,            // reference to matches.id
  board_size: 9,
  sgf: String,               // SGF string or S3 reference
  moves: [
    {
      number: 1,
      color: "B",
      position: [x, y],
      policy: [ ... ],       // optional: move probability vector
      value: 0.42            // optional: win prob after move
    },
    ...
  ],
  analysis: {
    winrate_curve: [0.5, 0.55, ...],
    key_mistakes: [
      { move_number: 32, loss: 0.12, recommendation: "play (2,3)" }
    ],
    comments: [
      { move_number: 10, text: "Joseki deviation" }
    ]
  },
  created_at: ISODate()
}

// premium_reports collection
{
  _id: ObjectId(),
  match_id: UUID,
  feature: "analysis" | "hint" | "review",
  summary: String,
  details: Object,         // JSON structure specific to feature
  coins_spent: Number,
  created_at: ISODate()
}

// ai_logs collection (debug)
{
  _id: ObjectId(),
  match_id: UUID,
  move_number: Number,
  engine: "MCTS" | "Minimax",
  config: {
    depth: 4,
    playouts: 5000,
    time_taken_ms: 3200
  },
  stats: {
    best_move: [x, y],
    principal_variation: [[x1, y1], [x2, y2]],
    visit_distribution: [
      { move: [x, y], visits: 1200, win_rate: 0.58 },
      ...
    ]
  },
  timestamp: ISODate()
}
```

Indexes: `games.match_id`, `premium_reports.match_id`, `ai_logs.match_id`.

---

## 6. Integration with AI Engines

- The backend loads the `gogame_py` module (pybind11) inside service layer.
- `match_service` orchestrates:
  1. Create `Board` from request or stored SGF.
  2. Use `AIPlayer.select_move(level)` for AI decisions.
  3. For premium features, call `mcts_result` or `minimax_result` for explanations, or `SelfPlayTrainer` for offline training.
- Backpressure: long-running analysis executed via background tasks to avoid blocking HTTP request (use FastAPI `BackgroundTasks` initially; migrate to Celery if needed).

---

## 7. API Design Overview

### Auth & Users
- `POST /auth/register` — create account (username, email, password).
- `POST /auth/login` — returns access + refresh tokens.
- `POST /auth/logout` — revoke refresh token.
- `POST /auth/refresh` — rotate access token.
- `GET /users/me` — profile + coin balance.
- `PATCH /users/me` — update display name, avatar, preferences.
- `GET /users/{id}` — public profile.

### Matches
- `POST /matches/pvp` — create PvP lobby.
- `POST /matches/pvp/{id}/join` — join existing lobby via code.
- `POST /matches/ai` — start AI match with level/board size.
- `POST /matches/{id}/move` — submit move (payload: x, y, move_number).
- `POST /matches/{id}/pass` — pass.
- `POST /matches/{id}/resign` — resign.
- `GET /matches/{id}` — metadata + current board state (compressed format).
- `GET /matches/{id}/sgf` — download SGF/JSON record.
- `GET /matches/history` — paginated list for user.
- `GET /matches/{id}/analysis` — fetch premium report (if completed).

### Premium / Coins
- `GET /coins/balance` — coin balance, daily bonus status.
- `POST /coins/purchase` — stub, adds coins (simulate payment).
- `GET /coins/history` — transaction history.
- `POST /premium/hint` — immediate top-k moves (spend coins).
- `POST /premium/analysis` — deep analysis (async, returns request ID).
- `POST /premium/review` — game review (async).
- `GET /premium/requests/{id}` — status/result of async request.

### ML / Admin
- `POST /ml/train` — trigger training iteration (admin only).
- `GET /ml/models` — list model versions.
- `POST /ml/models/promote` — set active model.

### Health & Misc
- `GET /health` — DB connectivity.
- `GET /config` — client dynamic config (e.g., AI levels, premium prices).

Endpoints return Pydantic schema; async premium operations respond with HTTP 202 and clients poll `/premium/requests/{id}` or subscribe to WebSocket updates.

#### Example Pydantic Schemas

```python
class RegisterRequest(BaseModel):
    username: constr(min_length=3, max_length=32)
    email: EmailStr
    password: constr(min_length=8)

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class MatchCreateAIRequest(BaseModel):
    level: conint(ge=1, le=4)
    board_size: conint(ge=9, le=19)

class MoveRequest(BaseModel):
    x: conint(ge=0)
    y: conint(ge=0)
    move_number: int

class PremiumHintRequest(BaseModel):
    match_id: UUID
    top_k: conint(ge=1, le=5) = 3
```

---

## 8. Security Considerations

- Use HTTPS (behind reverse proxy e.g. Nginx).
- JWT Access token (15 min), refresh token (7 days) stored server-side (refresh_tokens table).
- Rate limit login/registration via Redis or simple in-memory limiter (future).
- Validate SGF uploads to prevent injection.
- Strict coin transactions with ACID transactions in PostgreSQL.
- Logging of premium usage for audit.

---

## 9. Deployment Notes

- Docker Compose setup: `fastapi`, `postgres`, `mongo`, `minio`, optional `redis`.
- Environment variables managed via `.env` file read by `config.py`.
- Use Alembic migrations for schema updates.
- **Database Setup:** Có 2 cách setup database:
  1. **Tự động:** Sử dụng script Python `backend/scripts/setup_database.py` (khuyến nghị)
  2. **Thủ công:** Sử dụng SQL scripts trong `backend/scripts/`:
     - `database_schema.sql` - Tạo toàn bộ schema
     - `database_sample_data.sql` - Insert dữ liệu mẫu
     - `database_queries.sql` - Các query hữu ích
     - Xem `backend/scripts/README_SQL.md` để biết chi tiết
- Seed data script for admin user & sample matches (xem `database_sample_data.sql`).
- Monitoring: integrate with Prometheus via `/metrics` or use FastAPI middleware.

---

## 10. Implementation Status

✅ **Completed:**
1. FastAPI project structure under `backend/` - ✅ Done
2. SQLAlchemy models + Alembic migrations - ✅ Done
3. Auth & user endpoints with JWT - ✅ Done
4. Database schema scripts (SQL) - ✅ Done
5. Matchmaking system with ready status - ✅ Done
6. Match endpoints (PvP, PvAI) - ✅ Done
7. Premium feature endpoints (hint, analysis) - ✅ Done
8. Coin system and transactions - ✅ Done

🔄 **In Progress / Future:**
- WebSocket support for real-time match updates
- Celery workers for heavy ML jobs
- Model version management (`model_versions` table)
- Deployment automation (CI/CD)
- Comprehensive integration tests

## 11. Database Scripts Reference

Tất cả các script SQL để quản lý database được đặt trong `backend/scripts/`:

- **`database_schema.sql`** - Script chính để tạo toàn bộ schema
- **`database_drop.sql`** - Xóa database (⚠️ Cẩn thận!)
- **`database_reset.sql`** - Reset dữ liệu (giữ schema)
- **`database_backup.sql`** - Hướng dẫn backup
- **`database_sample_data.sql`** - Insert dữ liệu mẫu
- **`database_queries.sql`** - Các query hữu ích
- **`README_SQL.md`** - Tài liệu chi tiết về các script

Xem `backend/scripts/README_SQL.md` để biết hướng dẫn sử dụng đầy đủ.

---

This backend design provides a solid foundation for both core gameplay services and premium ML-powered features described in SystemSpec. Future iterations can add WebSocket matchmaking, Celery workers for heavy ML jobs, and deployment automation (CI/CD).

