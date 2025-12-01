# Hệ thống Tiền tệ và Premium

## Tổng quan

Hệ thống tiền tệ và Premium cho phép người dùng:
- Mua và sử dụng coins để mở khóa các tính năng premium
- Đăng ký premium subscription để nhận ưu đãi
- Kiếm coins miễn phí từ các hoạt động trong game

## Cấu trúc

### 1. Models

#### `PremiumSubscription`
Lưu thông tin premium subscription của user:
- `user_id`: ID của user (unique)
- `plan`: "monthly" hoặc "yearly"
- `status`: "active", "expired", "cancelled"
- `started_at`: Thời gian bắt đầu
- `expires_at`: Thời gian hết hạn
- `cancelled_at`: Thời gian hủy (nếu có)

#### `CoinTransaction`
Lưu lịch sử giao dịch coins:
- `user_id`: ID của user
- `amount`: Số coins (+ hoặc -)
- `type`: "purchase", "earn", "spend"
- `source`: Nguồn gốc (daily_login, package_starter, premium_hint, etc.)

### 2. Services

#### `CoinService`
Quản lý coins:
- `get_balance()`: Lấy số dư và trạng thái daily bonus
- `list_transactions()`: Lấy lịch sử giao dịch
- `add_transaction()`: Thêm transaction
- `claim_daily_bonus()`: Nhận daily bonus
- `earn_coins()`: Kiếm coins từ các hành động

#### `PaymentService`
Xử lý thanh toán (mock implementation):
- `purchase_coins()`: Mua coins từ package
- `subscribe_premium()`: Đăng ký premium subscription
- `cancel_subscription()`: Hủy subscription
- `get_subscription_status()`: Lấy trạng thái subscription

#### `PremiumService`
Tính năng premium (đã có sẵn):
- `create_hint()`: Gợi ý nước đi (10 coins)
- `create_analysis()`: Phân tích ván cờ (20 coins)
- `create_review()`: Review ván cờ (30 coins)
- `_is_premium_user()`: Kiểm tra user có premium không

### 3. API Endpoints

#### Coins

- `GET /coins/balance` - Lấy số dư coins
- `GET /coins/history` - Lấy lịch sử giao dịch
- `GET /coins/packages` - Lấy danh sách gói coin
- `POST /coins/purchase` - Mua coins
- `POST /coins/daily-bonus` - Nhận daily bonus
- `POST /coins/earn` - Kiếm coins từ hành động

#### Premium

- `GET /premium/subscription/status` - Lấy trạng thái subscription
- `GET /premium/subscription/plans` - Lấy danh sách gói subscription
- `POST /premium/subscription/subscribe` - Đăng ký premium
- `POST /premium/subscription/cancel` - Hủy subscription
- `POST /premium/hint` - Gợi ý nước đi (đã có)
- `POST /premium/analysis` - Phân tích ván cờ (đã có)
- `POST /premium/review` - Review ván cờ (đã có)

## Coin Packages

| Package ID | Coins | Bonus | Price (USD) |
|------------|-------|-------|-------------|
| starter    | 100   | 0     | $0.99       |
| basic      | 500   | 0     | $3.99       |
| standard   | 1200  | 200   | $8.99       |
| premium    | 3000  | 1500  | $19.99      |
| ultimate   | 10000 | 10000 | $49.99      |

## Premium Plans

| Plan    | Duration | Price (USD) | Bonus Coins |
|---------|----------|-------------|-------------|
| monthly | 30 days  | $4.99       | 500         |
| yearly  | 365 days | $49.99      | 6000        |

## Coin Earning Rules

| Action        | Coins |
|---------------|-------|
| daily_login   | 10    |
| complete_game | 5     |
| win_game      | 10    |
| rank_up       | 50    |
| achievement   | 20    |
| watch_ad      | 5     |

## Premium Features Cost

| Feature   | Cost (Coins) |
|-----------|--------------|
| Hint      | 10           |
| Analysis  | 20           |
| Review    | 30           |

## Usage Examples

### Mua coins

```python
POST /coins/purchase
{
    "package_id": "starter",
    "payment_token": "mock_token_123"  # Optional trong mock
}
```

### Đăng ký premium

```python
POST /premium/subscription/subscribe
{
    "plan": "monthly",
    "payment_token": "mock_token_123"  # Optional trong mock
}
```

### Nhận daily bonus

```python
POST /coins/daily-bonus
```

### Kiếm coins từ hành động

```python
POST /coins/earn
{
    "action": "win_game"
}
```

## Migration

Chạy migration để tạo bảng `premium_subscriptions`:

```bash
cd backend
alembic upgrade head
```

Hoặc nếu migration chưa được apply:

```bash
alembic revision --autogenerate -m "add_premium_subscription_table"
alembic upgrade head
```

## Payment Gateway Integration

Hiện tại `PaymentService` là mock implementation. Để tích hợp payment gateway thật:

1. Cập nhật `PaymentService._verify_payment()` để gọi API của payment gateway
2. Tích hợp với:
   - Stripe (quốc tế)
   - PayPal (quốc tế)
   - VNPay (Việt Nam)
   - MoMo (Việt Nam)
   - etc.

3. Lưu payment transaction ID để tracking

## Premium User Benefits

Premium users có thể:
- Giảm cost cho premium features (có thể config)
- Nhận bonus coins khi đăng ký
- Ưu tiên trong matchmaking (có thể thêm sau)
- Custom themes/avatars (có thể thêm sau)

## Future Enhancements

- [ ] Tích hợp payment gateway thật
- [ ] Premium user benefits mở rộng
- [ ] Referral system (kiếm coins khi giới thiệu)
- [ ] Achievement system với coin rewards
- [ ] Leaderboard với coin prizes
- [ ] Gift coins giữa users

