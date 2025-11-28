# Frontend Development Guide

## Overview

Frontend được xây dựng bằng **PyQt6**, một cross-platform GUI framework cho Python. Frontend kết nối với backend FastAPI qua REST API.

## Technology Stack

- **PyQt6**: Desktop GUI framework
- **httpx**: Async HTTP client cho API calls
- **python-dotenv**: Environment configuration

## Project Structure

```
frontend/
├── main.py                 # Entry point
├── app/
│   ├── __init__.py
│   ├── main_window.py      # Main application window
│   ├── api/
│   │   └── client.py       # Backend API client
│   ├── widgets/
│   │   ├── board_widget.py  # Go board visualization
│   │   ├── game_controls.py  # Game control buttons
│   │   ├── match_list.py    # Match history list
│   │   └── stats_panel.py   # Statistics panel
│   └── dialogs/
│       ├── login_dialog.py  # Login/Register dialog
│       └── match_dialog.py # Create match dialog
├── requirements.txt
├── env.example
└── README.md
```

## Features Implemented

### ✅ Core Features

1. **Authentication**
   - Login/Register dialog
   - JWT token management
   - Auto-login on startup

2. **Go Board Visualization**
   - 9x9 và 19x19 board support
   - Stone placement (Black/White)
   - Last move highlighting
   - Hover indicators
   - Hint moves visualization
   - Star points (hoshi) rendering

3. **Game Controls**
   - Move submission (click on board)
   - Pass turn
   - Resign match
   - Premium features (Hint, Analysis, Review)

4. **Match Management**
   - Create AI match (levels 1-4)
   - Create PvP match
   - Join PvP match
   - Match history viewer
   - Match replay

5. **Statistics Dashboard**
   - Elo rating display
   - Win/Loss statistics
   - Win rate calculation
   - Leaderboard (future)

## Setup & Installation

### 1. Install Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `env.example` to `.env`:

```env
BACKEND_URL=http://localhost:8000
```

### 3. Run Application

**Windows:**
```bash
python main.py
# hoặc
run.bat
```

**Linux/Mac:**
```bash
python3 main.py
# hoặc
chmod +x run.sh
./run.sh
```

## Usage

### Starting the Application

1. **Start Backend Server** (if not running):
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Start Frontend**:
   ```bash
   cd frontend
   python main.py
   ```

### Playing a Game

1. **Login/Register**: Application sẽ hiển thị login dialog khi khởi động
2. **Create Match**: Click "New Match" → Chọn AI level hoặc PvP → Chọn board size
3. **Play**: Click trên board để đặt quân cờ
4. **Controls**: 
   - **Pass**: Bỏ lượt
   - **Resign**: Đầu hàng
   - **Hint**: Gợi ý nước đi (tốn coins)
   - **Analysis**: Phân tích vị trí (tốn coins)
   - **Review**: Review toàn bộ ván đấu (tốn coins)

### Viewing Statistics

- Statistics panel hiển thị:
  - Elo rating
  - Total matches
  - Wins/Losses
  - Win rate

### Match History

- Click vào match trong history list để xem replay
- Board sẽ load lại tất cả moves của match đó

## API Integration

Frontend sử dụng `APIClient` class để gọi backend API:

```python
from app.api.client import APIClient

client = APIClient()
await client.login("username", "password")
await client.create_ai_match(level=1, board_size=9)
await client.submit_move(match_id, x=3, y=3, move_number=1, color="B")
```

## Known Issues & Limitations

1. **Async/Await với PyQt**: 
   - PyQt không có event loop riêng
   - Sử dụng `asyncio.get_event_loop()` với fallback
   - Có thể cần cải thiện async handling

2. **Error Handling**:
   - Basic error handling implemented
   - Cần thêm retry logic cho network errors

3. **UI Polish**:
   - Basic styling
   - Có thể cải thiện với custom stylesheets

4. **Real-time Updates**:
   - Hiện tại dùng polling (timer mỗi 2 giây)
   - Có thể upgrade lên WebSocket cho real-time

## Future Improvements

1. **WebSocket Support**: Real-time game updates
2. **Custom Themes**: Dark/Light mode
3. **Sound Effects**: Move sounds, game end sounds
4. **Animation**: Smooth stone placement animations
5. **Undo/Redo**: Move history navigation
6. **Analysis Panel**: Detailed move analysis display
7. **Leaderboard UI**: Full leaderboard viewer
8. **Settings**: User preferences, board appearance

## Testing

Để test frontend:

1. **Manual Testing**:
   - Start backend server
   - Run frontend
   - Test tất cả features

2. **Integration Testing**:
   - Test với backend API
   - Verify data flow
   - Check error handling

## Troubleshooting

### Import Errors

Nếu gặp import errors, đảm bảo:
- Đang ở đúng directory (`frontend/`)
- Python path includes `frontend/`
- Dependencies đã install

### Connection Errors

Nếu không kết nối được backend:
- Check `BACKEND_URL` trong `.env`
- Đảm bảo backend server đang chạy
- Check firewall/network settings

### PyQt6 Installation Issues

Nếu PyQt6 không install được:
- **Windows**: `pip install PyQt6`
- **Linux**: Có thể cần `sudo apt-get install python3-pyqt6`
- **Mac**: `brew install pyqt6`

## Development Notes

- Frontend code sử dụng type hints
- Async/await pattern cho API calls
- Signal/slot pattern cho PyQt events
- Separation of concerns: API client, widgets, dialogs

