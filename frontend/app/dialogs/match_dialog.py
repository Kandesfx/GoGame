"""Create match dialog."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QSpinBox

from ..styles import BUTTON_STYLE, DIALOG_STYLE, FORM_STYLE


class MatchDialog(QDialog):
    """Dialog để tạo match."""

    match_created = pyqtSignal(str, int, int)  # match_type, level/board_size, board_size

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Match")
        self.setMinimumWidth(300)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QFormLayout()
        layout.setSpacing(15)
        self.setLayout(layout)
        self.setStyleSheet(FORM_STYLE)

        # Match type
        self.match_type = QComboBox()
        self.match_type.addItems(["AI", "PvP"])
        layout.addRow("Match Type:", self.match_type)

        # AI Level (only for AI matches)
        self.ai_level = QSpinBox()
        self.ai_level.setMinimum(1)
        self.ai_level.setMaximum(4)
        self.ai_level.setValue(1)
        layout.addRow("AI Level:", self.ai_level)

        # Board size
        self.board_size = QSpinBox()
        self.board_size.setMinimum(9)
        self.board_size.setMaximum(19)
        self.board_size.setValue(9)
        layout.addRow("Board Size:", self.board_size)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Show/hide AI level based on match type
        self.match_type.currentTextChanged.connect(self._on_type_changed)
        self._on_type_changed("AI")

    def _on_type_changed(self, text: str) -> None:
        """Handle match type change."""
        self.ai_level.setVisible(text == "AI")

    def _on_ok(self) -> None:
        """Handle OK button."""
        match_type = self.match_type.currentText().lower()
        level = self.ai_level.value() if match_type == "ai" else 0
        board_size = self.board_size.value()
        self.match_created.emit(match_type, level, board_size)
        self.accept()

