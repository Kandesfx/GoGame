"""Match history list widget."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QListWidget, QListWidgetItem

from ..styles import PANEL_STYLE


class MatchListWidget(QListWidget):
    """Widget hiển thị danh sách matches."""

    match_selected = pyqtSignal(UUID)  # match_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(PANEL_STYLE)
        self.itemClicked.connect(self._on_item_clicked)
        self._matches: dict[str, UUID] = {}  # item_id -> match_id

    def add_match(self, match_id: UUID, info: str) -> None:
        """Add match to list."""
        item = QListWidgetItem(info)
        item_id = str(match_id)
        self._matches[item_id] = match_id
        item.setData(QListWidgetItem.ItemDataRole.UserRole, item_id)
        self.addItem(item)

    def clear_matches(self) -> None:
        """Clear all matches."""
        self.clear()
        self._matches.clear()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle item click."""
        item_id = item.data(QListWidgetItem.ItemDataRole.UserRole)
        if item_id and item_id in self._matches:
            self.match_selected.emit(self._matches[item_id])

