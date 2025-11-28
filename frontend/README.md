# GoGame Frontend

Desktop application cho GoGame sử dụng PyQt6.

## Setup


### 1. Install Dependencies

```bash
# Windows
pip install -r requirements.txt

# Linux/Mac
pip3 install -r requirements.txt
```

### 2. Configuration

Copy `env.example` to `.env` và cấu hình:

```env
BACKEND_URL=http://localhost:8000
```

### 3. Run Application

```bash
python main.py
```

## Features

- ✅ Go Board Visualization (9x9, 19x19)
- ✅ Game Controls (Move, Pass, Resign)
- ✅ Match Creation (AI, PvP)
- ✅ Match History Viewer
- ✅ Statistics Dashboard
- ✅ Authentication (Login/Register)
- ✅ Premium Features UI (Hint, Analysis, Review)

## Project Structure

```
frontend/
├── main.py                 # Entry point
├── app/
│   ├── __init__.py
│   ├── main_window.py      # Main application window
│   ├── widgets/
│   │   ├── board_widget.py  # Go board visualization
│   │   ├── game_controls.py  # Game control buttons
│   │   ├── match_list.py    # Match history list
│   │   └── stats_panel.py   # Statistics panel
│   ├── dialogs/
│   │   ├── login_dialog.py  # Login/Register dialog
│   │   ├── match_dialog.py # Create match dialog
│   │   └── premium_dialog.py # Premium features dialog
│   └── api/
│       └── client.py       # Backend API client
├── resources/              # Images, icons
└── requirements.txt
```

