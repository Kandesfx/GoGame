"""Custom stylesheets cho GoGame UI."""

# Modern color scheme
COLORS = {
    "primary": "#2C3E50",      # Dark blue-gray
    "secondary": "#3498DB",    # Blue
    "success": "#27AE60",      # Green
    "danger": "#E74C3C",       # Red
    "warning": "#F39C12",     # Orange
    "info": "#3498DB",        # Blue
    "light": "#ECF0F1",        # Light gray
    "dark": "#34495E",         # Dark gray
    "board_bg": "#DCB35C",    # Wood color
    "board_line": "#8B4513",  # Brown
    "stone_black": "#1A1A1A", # Black
    "stone_white": "#F5F5F5", # White
    "hover": "#3498DB",       # Blue hover
    "hint": "#2ECC71",        # Green hint
}

# Main window stylesheet
MAIN_WINDOW_STYLE = f"""
QMainWindow {{
    background-color: {COLORS["light"]};
    color: {COLORS["primary"]};
}}

QMenuBar {{
    background-color: {COLORS["primary"]};
    color: white;
    padding: 4px;
    border: none;
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 8px 16px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS["secondary"]};
}}

QMenu {{
    background-color: white;
    border: 1px solid {COLORS["dark"]};
    border-radius: 4px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {COLORS["secondary"]};
    color: white;
}}

QStatusBar {{
    background-color: {COLORS["primary"]};
    color: white;
    padding: 4px;
}}
"""

# Button stylesheet
BUTTON_STYLE = f"""
QPushButton {{
    background-color: {COLORS["secondary"]};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 14px;
    min-height: 30px;
}}

QPushButton:hover {{
    background-color: #2980B9;
}}

QPushButton:pressed {{
    background-color: #21618C;
}}

QPushButton:disabled {{
    background-color: {COLORS["light"]};
    color: #95A5A6;
}}
"""

# Premium button style (different color)
PREMIUM_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {COLORS["warning"]};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 14px;
    min-height: 30px;
}}

QPushButton:hover {{
    background-color: #E67E22;
}}

QPushButton:pressed {{
    background-color: #D35400;
}}
"""

# Danger button (Resign)
DANGER_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {COLORS["danger"]};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 14px;
    min-height: 30px;
}}

QPushButton:hover {{
    background-color: #C0392B;
}}

QPushButton:pressed {{
    background-color: #A93226;
}}
"""

# Panel stylesheet
PANEL_STYLE = f"""
QWidget {{
    background-color: white;
    border-radius: 8px;
    padding: 10px;
}}

QLabel {{
    color: {COLORS["primary"]};
    font-size: 14px;
    font-weight: bold;
}}

QListWidget {{
    background-color: white;
    border: 2px solid {COLORS["light"]};
    border-radius: 6px;
    padding: 5px;
    font-size: 13px;
}}

QListWidget::item {{
    padding: 8px;
    border-radius: 4px;
    margin: 2px;
}}

QListWidget::item:selected {{
    background-color: {COLORS["secondary"]};
    color: white;
}}

QListWidget::item:hover {{
    background-color: {COLORS["light"]};
}}
"""

# Form layout style
FORM_STYLE = f"""
QLabel {{
    color: {COLORS["primary"]};
    font-weight: bold;
    font-size: 13px;
}}

QLineEdit {{
    border: 2px solid {COLORS["light"]};
    border-radius: 4px;
    padding: 8px;
    font-size: 13px;
    background-color: white;
}}

QLineEdit:focus {{
    border-color: {COLORS["secondary"]};
}}

QSpinBox {{
    border: 2px solid {COLORS["light"]};
    border-radius: 4px;
    padding: 8px;
    font-size: 13px;
    background-color: white;
}}

QSpinBox:focus {{
    border-color: {COLORS["secondary"]};
}}

QComboBox {{
    border: 2px solid {COLORS["light"]};
    border-radius: 4px;
    padding: 8px;
    font-size: 13px;
    background-color: white;
}}

QComboBox:focus {{
    border-color: {COLORS["secondary"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {COLORS["primary"]};
    margin-right: 5px;
}}
"""

# Dialog style
DIALOG_STYLE = f"""
QDialog {{
    background-color: white;
    border-radius: 8px;
}}

QTabWidget::pane {{
    border: 2px solid {COLORS["light"]};
    border-radius: 6px;
    background-color: white;
}}

QTabBar::tab {{
    background-color: {COLORS["light"]};
    color: {COLORS["primary"]};
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: bold;
}}

QTabBar::tab:selected {{
    background-color: {COLORS["secondary"]};
    color: white;
}}

QTabBar::tab:hover {{
    background-color: {COLORS["dark"]};
    color: white;
}}
"""

# Statistics panel style
STATS_PANEL_STYLE = f"""
QWidget {{
    background-color: white;
    border: 2px solid {COLORS["light"]};
    border-radius: 8px;
    padding: 15px;
}}

QLabel {{
    color: {COLORS["primary"]};
    font-size: 14px;
}}

QFormLayout QLabel {{
    font-weight: normal;
    color: {COLORS["dark"]};
}}
"""

