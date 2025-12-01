"""Statistics panel widget."""

from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget

from ..styles import STATS_PANEL_STYLE


class StatsPanelWidget(QWidget):
    """Widget hiển thị statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STATS_PANEL_STYLE)

        layout = QFormLayout()
        layout.setSpacing(10)
        self.setLayout(layout)

        # Labels
        self.elo_label = QLabel("1500")
        self.matches_label = QLabel("0")
        self.wins_label = QLabel("0")
        self.losses_label = QLabel("0")
        self.win_rate_label = QLabel("0.0%")

        layout.addRow("Elo Rating:", self.elo_label)
        layout.addRow("Total Matches:", self.matches_label)
        layout.addRow("Wins:", self.wins_label)
        layout.addRow("Losses:", self.losses_label)
        layout.addRow("Win Rate:", self.win_rate_label)

    def update_stats(self, stats: dict) -> None:
        """Update statistics display."""
        self.elo_label.setText(str(stats.get("elo_rating", 1500)))
        self.matches_label.setText(str(stats.get("total_matches", 0)))
        self.wins_label.setText(str(stats.get("wins", 0)))
        self.losses_label.setText(str(stats.get("losses", 0)))
        win_rate = stats.get("win_rate", 0.0)
        self.win_rate_label.setText(f"{win_rate:.1f}%")

