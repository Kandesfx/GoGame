# 🏆 Hệ Thống ELO và Xếp Hạng (Ranking System)

## 📋 Tổng Quan

Hệ thống sử dụng **ELO rating** để đánh giá trình độ người chơi và xếp hạng họ trên bảng xếp hạng (leaderboard).

---

## 🎯 ELO Rating

### Khởi Tạo
- **ELO ban đầu**: `1500` điểm (cho tất cả người chơi mới)
- **Lưu trữ**: Trong bảng `users.elo_rating` (PostgreSQL)

### Công Thức Tính ELO

#### 1. Expected Score (Điểm Kỳ Vọng)
```
Expected Score = 1 / (1 + 10^((opponent_rating - your_rating) / 400))
```

**Ví dụ:**
- Bạn có ELO: `1500`
- Đối thủ có ELO: `1600`
- Expected Score = `1 / (1 + 10^((1600-1500)/400))` = `1 / (1 + 10^0.25)` ≈ `0.36`
- → Bạn có **36%** cơ hội thắng (theo lý thuyết)

#### 2. ELO Change (Thay Đổi Điểm)
```
ELO Change = K_FACTOR × (Actual Score - Expected Score)
```

**Trong đó:**
- **K_FACTOR** = `32` (hệ số chuẩn, quyết định tốc độ thay đổi ELO)
- **Actual Score**:
  - `1.0` = Thắng
  - `0.5` = Hòa
  - `0.0` = Thua

**Ví dụ:**
- Bạn có ELO: `1500`, Expected Score: `0.36`
- Bạn **thắng** (Actual Score = `1.0`)
- ELO Change = `32 × (1.0 - 0.36)` = `32 × 0.64` = **+20 điểm**
- ELO mới = `1500 + 20` = **1520**

**Nếu bạn thua:**
- ELO Change = `32 × (0.0 - 0.36)` = `32 × (-0.36)` = **-12 điểm**
- ELO mới = `1500 - 12` = **1488**

### Cập Nhật ELO

ELO chỉ được cập nhật khi:
1. ✅ **Match kết thúc** (có `result`)
2. ✅ **Match là PvP** (không phải AI match)
3. ✅ **Có đủ 2 người chơi** (black_player và white_player)

**ELO KHÔNG được cập nhật khi:**
- ❌ Match với AI (AI matches không ảnh hưởng ELO)
- ❌ Match chưa kết thúc
- ❌ Match không có result

### Giới Hạn
- **ELO tối thiểu**: `0` (không cho phép ELO âm)
- **ELO tối đa**: Không giới hạn (có thể tăng vô hạn)

---

## 📊 Xếp Hạng (Rank)

### Cách Tính Rank
**Rank = Vị trí trên Leaderboard** (xếp theo ELO giảm dần)

```
Rank 1 = Người có ELO cao nhất
Rank 2 = Người có ELO cao thứ 2
...
```

### Leaderboard
- **Sắp xếp**: Theo `elo_rating` **giảm dần** (DESC)
- **Giới hạn**: Top 100 người chơi (mặc định)
- **Thông tin hiển thị**:
  - Rank (vị trí)
  - Username
  - Display Name
  - ELO Rating
  - Total Matches
  - Win Rate

### API Endpoint
```
GET /statistics/leaderboard?limit=100
```

---

## 📈 Ví Dụ Tính Toán

### Scenario 1: Thắng Đối Thủ Mạnh Hơn
```
Bạn: 1500 ELO
Đối thủ: 1700 ELO

Expected Score = 1 / (1 + 10^((1700-1500)/400))
              = 1 / (1 + 10^0.5)
              = 1 / (1 + 3.16)
              ≈ 0.24 (24% cơ hội thắng)

Bạn THẮNG:
ELO Change = 32 × (1.0 - 0.24) = 32 × 0.76 = +24 điểm
ELO mới = 1500 + 24 = 1524

Đối thủ THUA:
ELO Change = 32 × (0.0 - 0.76) = 32 × (-0.76) = -24 điểm
ELO mới = 1700 - 24 = 1676
```

### Scenario 2: Thắng Đối Thủ Yếu Hơn
```
Bạn: 1500 ELO
Đối thủ: 1300 ELO

Expected Score = 1 / (1 + 10^((1300-1500)/400))
              = 1 / (1 + 10^(-0.5))
              = 1 / (1 + 0.32)
              ≈ 0.76 (76% cơ hội thắng)

Bạn THẮNG:
ELO Change = 32 × (1.0 - 0.76) = 32 × 0.24 = +8 điểm
ELO mới = 1500 + 8 = 1508

Đối thủ THUA:
ELO Change = 32 × (0.0 - 0.24) = 32 × (-0.24) = -8 điểm
ELO mới = 1300 - 8 = 1292
```

### Scenario 3: Hòa
```
Bạn: 1500 ELO
Đối thủ: 1500 ELO

Expected Score = 1 / (1 + 10^((1500-1500)/400))
              = 1 / (1 + 10^0)
              = 1 / (1 + 1)
              = 0.5 (50% cơ hội thắng)

HÒA:
ELO Change = 32 × (0.5 - 0.5) = 32 × 0 = 0 điểm
ELO không đổi = 1500
```

---

## 🔧 Implementation Details

### Backend Code

**File**: `backend/app/services/statistics_service.py`

```python
# Constants
K_FACTOR = 32
INITIAL_RATING = 1500

def calculate_expected_score(rating_a: int, rating_b: int) -> float:
    """Tính expected score cho player A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

def calculate_elo_change(rating: int, opponent_rating: int, actual_score: float) -> int:
    """Tính Elo rating change."""
    expected_score = calculate_expected_score(rating, opponent_rating)
    change = int(K_FACTOR * (actual_score - expected_score))
    return change
```

### Khi Nào ELO Được Cập Nhật?

**File**: `backend/app/services/match_service.py`

```python
# Update Elo ratings nếu match kết thúc và là PvP
if match.result and not match.ai_level:
    stats_service = StatisticsService(self.db)
    stats_service.update_elo_ratings(match)
```

### Database Schema

**File**: `backend/app/models/sql/user.py`

```python
class User(Base):
    elo_rating: Mapped[int] = mapped_column(Integer, default=1500)
```

---

## 📊 Phân Loại ELO (Tham Khảo)

Mặc dù hệ thống không có rank cố định, nhưng có thể phân loại theo ELO:

| ELO Range | Mô Tả |
|-----------|-------|
| 0 - 1000 | Người mới bắt đầu |
| 1000 - 1300 | Người chơi nghiệp dư |
| 1300 - 1600 | Người chơi trung bình |
| 1600 - 1900 | Người chơi khá |
| 1900 - 2200 | Người chơi giỏi |
| 2200+ | Người chơi xuất sắc |

---

## ⚠️ Lưu Ý Quan Trọng

1. **AI Matches KHÔNG ảnh hưởng ELO**
   - Chỉ PvP matches mới cập nhật ELO
   - Điều này đảm bảo ELO phản ánh trình độ thực tế giữa người với người

2. **ELO là Zero-Sum**
   - Tổng ELO của 2 người chơi sau match = Tổng ELO trước match
   - Nếu bạn +20, đối thủ sẽ -20 (hoặc ngược lại)

3. **K-Factor = 32**
   - Là giá trị chuẩn cho người chơi đã có kinh nghiệm
   - Có thể điều chỉnh để:
     - Tăng tốc độ thay đổi: K = 40, 50
     - Giảm tốc độ thay đổi: K = 24, 16

4. **Expected Score**
   - Chênh lệch 200 ELO = ~75% cơ hội thắng
   - Chênh lệch 400 ELO = ~91% cơ hội thắng
   - Chênh lệch 800 ELO = ~99% cơ hội thắng

---

## 🎮 Frontend Display

### Statistics Panel
- Hiển thị ELO hiện tại của user
- Cập nhật real-time sau mỗi PvP match

### Leaderboard
- Hiển thị top players theo ELO
- Rank được tính tự động từ vị trí trên leaderboard

---

## 🔮 Cải Thiện Tương Lai

1. **Provisional Rating**
   - Người chơi mới (< 20 matches) có K-factor cao hơn
   - Giúp ELO nhanh chóng ổn định

2. **Rank Tiers**
   - Bronze, Silver, Gold, Platinum, Diamond, Master, Grandmaster
   - Dựa trên ELO ranges

3. **Seasonal Rankings**
   - Reset ELO mỗi mùa
   - Lưu lịch sử ELO theo mùa

4. **ELO Decay**
   - Giảm ELO nếu không chơi trong thời gian dài
   - Khuyến khích người chơi hoạt động

---

## 📚 Tài Liệu Tham Khảo

- [ELO Rating System - Wikipedia](https://en.wikipedia.org/wiki/Elo_rating_system)
- [USCF Rating System](https://www.uschess.org/content/view/7327/131)
- [FIDE Rating Regulations](https://handbook.fide.com/chapter/B022017)

