# GoGame Frontend - ReactJS

Web frontend cho GoGame sử dụng ReactJS và Vite.

## Prerequisites

**Cần cài đặt Node.js trước!**

- Download từ: https://nodejs.org/ (LTS version)
- Verify: `node --version` và `npm --version`

**⚠️ Nếu Git Bash không nhận Node.js:**
- Xem `README_NODEJS_FIX.md` để fix PATH
- Hoặc dùng **Command Prompt/PowerShell** thay vì Git Bash
- Hoặc chạy script: `bash fix_nodejs_path.sh`

Xem `SETUP.md` để biết chi tiết cài đặt.

## Setup

### 1. Install Dependencies

```bash
npm install
# hoặc
yarn install
# hoặc
pnpm install
```

### 2. Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:
```env
VITE_API_URL=http://localhost:8000
```

### 3. Run Development Server

```bash
npm run dev
# hoặc
yarn dev
# hoặc
pnpm dev
```

Ứng dụng sẽ chạy tại `http://localhost:3000`

## Features

- ✅ Authentication (Login/Register)
- ✅ Go Board Visualization (9x9, 19x19)
- ✅ Game Controls (Move, Pass, Resign)
- ✅ Match Creation (AI, PvP)
- ✅ Match History Viewer
- ✅ Statistics Dashboard
- ✅ Premium Features UI (Hint, Analysis, Review)

## Project Structure

```
frontend-web/
├── src/
│   ├── components/          # React components
│   │   ├── Board.jsx        # Go board visualization
│   │   ├── GameControls.jsx # Game control buttons
│   │   ├── LoginDialog.jsx  # Login/Register dialog
│   │   ├── MainWindow.jsx   # Main application window
│   │   ├── MatchDialog.jsx  # Create match dialog
│   │   ├── MatchList.jsx    # Match history list
│   │   └── StatisticsPanel.jsx # Statistics panel
│   ├── contexts/            # React contexts
│   │   └── AuthContext.jsx  # Authentication context
│   ├── services/            # API services
│   │   └── api.js           # Axios API client
│   ├── App.jsx              # Main App component
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── package.json
├── vite.config.js
└── README.md
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
   cd frontend-web
   npm run dev
   ```

3. **Open Browser**: Navigate to `http://localhost:3000`

### Playing a Game

1. **Login/Register**: Enter credentials
2. **Create Match**: Click "New Match" → Select AI level or PvP → Select board size
3. **Play**: Click on board to place stones
4. **Controls**: 
   - **Pass**: Skip turn
   - **Resign**: Surrender
   - **Hint**: Get move suggestions (premium)
   - **Analysis**: Analyze position (premium)
   - **Review**: Review game (premium)

## Build for Production

```bash
npm run build
```

Output sẽ ở trong thư mục `dist/`

## Preview Production Build

```bash
npm run preview
```

## Technology Stack

- **React 18**: UI framework
- **Vite**: Build tool và dev server
- **Axios**: HTTP client
- **React Router**: Routing (future)
- **CSS3**: Styling với CSS variables

## API Integration

Frontend sử dụng Axios để gọi backend API:

```javascript
import api from './services/api'

// Login
await api.post('/auth/login', { username_or_email, password })

// Create match
await api.post('/matches/ai', { level: 1, board_size: 9 })

// Submit move
await api.post(`/matches/${matchId}/move`, { x, y, move_number, color })
```

## Development Notes

- Components sử dụng React Hooks (useState, useEffect)
- Context API cho authentication state
- Axios interceptors cho token management
- Responsive design với CSS Grid và Flexbox
- Modern CSS với variables cho theming

