"""Game control buttons widget."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..styles import BUTTON_STYLE, DANGER_BUTTON_STYLE, PREMIUM_BUTTON_STYLE


class GameControlsWidget(QWidget):
    """Widget chứa các nút điều khiển game."""

    pass_clicked = pyqtSignal()
    resign_clicked = pyqtSignal()
    hint_clicked = pyqtSignal()
    analysis_clicked = pyqtSignal()
    review_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Main controls
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        self.pass_btn = QPushButton("⏭ Pass")
        self.pass_btn.setStyleSheet(BUTTON_STYLE)
        self.resign_btn = QPushButton("🏳 Resign")
        self.resign_btn.setStyleSheet(DANGER_BUTTON_STYLE)
        main_layout.addWidget(self.pass_btn)
        main_layout.addWidget(self.resign_btn)
        layout.addLayout(main_layout)

        # Premium features
        premium_layout = QHBoxLayout()
        premium_layout.setSpacing(10)
        self.hint_btn = QPushButton("💡 Hint")
        self.hint_btn.setStyleSheet(PREMIUM_BUTTON_STYLE)
        self.analysis_btn = QPushButton("📊 Analysis")
        self.analysis_btn.setStyleSheet(PREMIUM_BUTTON_STYLE)
        self.review_btn = QPushButton("📝 Review")
        self.review_btn.setStyleSheet(PREMIUM_BUTTON_STYLE)
        premium_layout.addWidget(self.hint_btn)
        premium_layout.addWidget(self.analysis_btn)
        premium_layout.addWidget(self.review_btn)
        layout.addLayout(premium_layout)

        # Connect signals
        self.pass_btn.clicked.connect(self.pass_clicked.emit)
        self.resign_btn.clicked.connect(self.resign_clicked.emit)
        self.hint_btn.clicked.connect(self.hint_clicked.emit)
        self.analysis_btn.clicked.connect(self.analysis_clicked.emit)
        self.review_btn.clicked.connect(self.review_clicked.emit)

        # Add stretch
        layout.addStretch()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable all controls."""
        self.pass_btn.setEnabled(enabled)
        self.resign_btn.setEnabled(enabled)
        self.hint_btn.setEnabled(enabled)
        self.analysis_btn.setEnabled(enabled)
        self.review_btn.setEnabled(enabled)

