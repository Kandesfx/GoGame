"""Go board visualization widget."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class BoardWidget(QWidget):
    """Widget để hiển thị bàn cờ Go."""

    move_clicked = pyqtSignal(int, int)  # x, y

    def __init__(self, board_size: int = 9, parent=None):
        super().__init__(parent)
        self.board_size = board_size
        self.stones: dict[tuple[int, int], str] = {}  # (x, y) -> "B" or "W"
        self.last_move: Optional[tuple[int, int]] = None
        self.hint_moves: list[tuple[int, int, float]] = []  # (x, y, confidence)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self._hover_pos: Optional[tuple[int, int]] = None

    def set_board_size(self, size: int) -> None:
        """Set board size."""
        self.board_size = size
        self.update()

    def set_stone(self, x: int, y: int, color: str) -> None:
        """Set stone at position."""
        if color in ("B", "W"):
            self.stones[(x, y)] = color
            self.last_move = (x, y)
            self.update()

    def clear_stone(self, x: int, y: int) -> None:
        """Clear stone at position."""
        self.stones.pop((x, y), None)
        self.update()

    def clear_board(self) -> None:
        """Clear all stones."""
        self.stones.clear()
        self.last_move = None
        self.hint_moves = []
        self.update()

    def set_hints(self, hints: list[dict]) -> None:
        """Set hint moves."""
        self.hint_moves = []
        for hint in hints:
            move = hint.get("move")
            if move and len(move) == 2:
                confidence = hint.get("confidence", 0.5)
                self.hint_moves.append((move[0], move[1], confidence))
        self.update()

    def clear_hints(self) -> None:
        """Clear hint moves."""
        self.hint_moves = []
        self.update()

    def _get_grid_rect(self) -> QRect:
        """Get grid drawing rectangle."""
        margin = 20
        size = min(self.width(), self.height()) - 2 * margin
        x = (self.width() - size) // 2
        y = (self.height() - size) // 2
        return QRect(x, y, size, size)

    def _pos_to_grid(self, pos: QPoint) -> Optional[tuple[int, int]]:
        """Convert screen position to grid coordinates."""
        rect = self._get_grid_rect()
        if not rect.contains(pos):
            return None

        cell_size = rect.width() / (self.board_size - 1)
        x = round((pos.x() - rect.x()) / cell_size)
        y = round((pos.y() - rect.y()) / cell_size)

        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return (x, y)
        return None

    def _grid_to_pos(self, x: int, y: int) -> QPoint:
        """Convert grid coordinates to screen position."""
        rect = self._get_grid_rect()
        cell_size = rect.width() / (self.board_size - 1)
        px = rect.x() + x * cell_size
        py = rect.y() + y * cell_size
        return QPoint(int(px), int(py))

    def paintEvent(self, event):
        """Paint board."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self._get_grid_rect()
        cell_size = rect.width() / (self.board_size - 1)

        # Draw board background with gradient
        from PyQt6.QtGui import QLinearGradient
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor(220, 179, 92))  # Light wood
        gradient.setColorAt(1, QColor(180, 140, 70))  # Dark wood
        painter.fillRect(rect, gradient)

        # Draw grid lines with better color
        pen = QPen(QColor(139, 69, 19), 1.5)  # Brown
        painter.setPen(pen)

        for i in range(self.board_size):
            # Horizontal lines
            y = rect.y() + i * cell_size
            painter.drawLine(int(rect.x()), int(y), int(rect.x() + rect.width()), int(y))

            # Vertical lines
            x = rect.x() + i * cell_size
            painter.drawLine(int(x), int(rect.y()), int(x), int(rect.y() + rect.height()))

        # Draw star points (hoshi)
        if self.board_size >= 9:
            star_points = []
            if self.board_size == 9:
                star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
            elif self.board_size == 19:
                star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]

            pen.setWidth(4)
            painter.setPen(pen)
            for sx, sy in star_points:
                pos = self._grid_to_pos(sx, sy)
                painter.drawPoint(pos)

        # Draw hint moves with better visualization
        if self.hint_moves:
            for hx, hy, confidence in self.hint_moves:
                pos = self._grid_to_pos(hx, hy)
                alpha = int(100 + confidence * 155)  # 100-255
                
                # Outer ring
                painter.setBrush(Qt.BrushStyle.NoBrush)
                pen = QPen(QColor(46, 204, 113, alpha), 3)  # Green
                painter.setPen(pen)
                radius = int(cell_size * 0.35)
                painter.drawEllipse(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
                
                # Inner circle
                painter.setBrush(QColor(46, 204, 113, alpha // 2))
                painter.setPen(Qt.PenStyle.NoPen)
                inner_radius = int(cell_size * 0.2)
                painter.drawEllipse(pos.x() - inner_radius, pos.y() - inner_radius, inner_radius * 2, inner_radius * 2)

        # Draw stones with shadow and gradient
        stone_radius = int(cell_size * 0.4)
        for (x, y), color in self.stones.items():
            pos = self._grid_to_pos(x, y)
            
            # Draw shadow
            shadow_offset = 2
            painter.setBrush(QColor(0, 0, 0, 80))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                pos.x() - stone_radius + shadow_offset,
                pos.y() - stone_radius + shadow_offset,
                stone_radius * 2,
                stone_radius * 2
            )
            
            # Draw stone with gradient
            from PyQt6.QtGui import QRadialGradient
            if color == "B":
                gradient = QRadialGradient(pos.x() - stone_radius * 0.3, pos.y() - stone_radius * 0.3, stone_radius * 2)
                gradient.setColorAt(0, QColor(60, 60, 60))  # Light black
                gradient.setColorAt(1, QColor(10, 10, 10))  # Dark black
                painter.setBrush(gradient)
                painter.setPen(QPen(QColor(30, 30, 30), 1.5))
            else:
                gradient = QRadialGradient(pos.x() - stone_radius * 0.3, pos.y() - stone_radius * 0.3, stone_radius * 2)
                gradient.setColorAt(0, QColor(255, 255, 255))  # White
                gradient.setColorAt(1, QColor(240, 240, 240))  # Light gray
                painter.setBrush(gradient)
                painter.setPen(QPen(QColor(200, 200, 200), 1.5))

            painter.drawEllipse(pos.x() - stone_radius, pos.y() - stone_radius, stone_radius * 2, stone_radius * 2)

        # Highlight last move with better visibility
        if self.last_move:
            lx, ly = self.last_move
            pos = self._grid_to_pos(lx, ly)
            
            # Outer ring
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(QColor(231, 76, 60, 200), 3)  # Red
            painter.setPen(pen)
            radius = int(cell_size * 0.3)
            painter.drawEllipse(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
            
            # Inner highlight
            painter.setBrush(QColor(231, 76, 60, 100))
            painter.setPen(Qt.PenStyle.NoPen)
            inner_radius = int(cell_size * 0.15)
            painter.drawEllipse(pos.x() - inner_radius, pos.y() - inner_radius, inner_radius * 2, inner_radius * 2)

        # Draw hover indicator with animation effect
        if self._hover_pos:
            hx, hy = self._hover_pos
            pos = self._grid_to_pos(hx, hy)
            
            # Outer ring (animated effect)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(QColor(52, 152, 219, 150), 2)  # Blue
            painter.setPen(pen)
            radius = int(cell_size * 0.4)
            painter.drawEllipse(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
            
            # Inner circle
            painter.setBrush(QColor(52, 152, 219, 80))
            painter.setPen(Qt.PenStyle.NoPen)
            inner_radius = int(cell_size * 0.25)
            painter.drawEllipse(pos.x() - inner_radius, pos.y() - inner_radius, inner_radius * 2, inner_radius * 2)

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        grid_pos = self._pos_to_grid(event.position().toPoint())
        if grid_pos:
            self._hover_pos = grid_pos
            self.update()
        else:
            self._hover_pos = None
            self.update()

    def mousePressEvent(self, event):
        """Handle mouse click."""
        if event.button() == Qt.MouseButton.LeftButton:
            grid_pos = self._pos_to_grid(event.position().toPoint())
            if grid_pos:
                x, y = grid_pos
                self.move_clicked.emit(x, y)

    def leaveEvent(self, event):
        """Handle mouse leave."""
        self._hover_pos = None
        self.update()

