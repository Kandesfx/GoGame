"""Main application window."""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import UUID

from PyQt6.QtCore import QTimer, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.api.client import APIClient
from app.dialogs.login_dialog import LoginDialog
from app.dialogs.match_dialog import MatchDialog
from app.styles import BUTTON_STYLE, MAIN_WINDOW_STYLE, PANEL_STYLE
from app.widgets.board_widget import BoardWidget
from app.widgets.game_controls import GameControlsWidget
from app.widgets.match_list import MatchListWidget
from app.widgets.stats_panel import StatsPanelWidget


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.api_client = APIClient()
        self.current_match_id: Optional[UUID] = None
        self.current_match_state: Optional[dict] = None
        self.current_player_color: Optional[str] = None
        self.move_number = 0

        self._init_ui()
        self._init_timer()

        # Show login dialog
        self._show_login()

    def _init_ui(self) -> None:
        """Initialize UI."""
        self.setWindowTitle("GoGame - 囲碁")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("📁 File")
        file_menu.addAction("🎮 New Match", self._on_new_match)
        file_menu.addAction("🚪 Logout", self._on_logout)
        file_menu.addSeparator()
        file_menu.addAction("❌ Exit", self.close)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Left panel: Board + Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Board
        self.board = BoardWidget(board_size=9)
        self.board.move_clicked.connect(self._on_board_click)
        left_layout.addWidget(self.board)

        # Game controls
        self.controls = GameControlsWidget()
        self.controls.pass_clicked.connect(self._on_pass)
        self.controls.resign_clicked.connect(self._on_resign)
        self.controls.hint_clicked.connect(self._on_hint)
        self.controls.analysis_clicked.connect(self._on_analysis)
        self.controls.review_clicked.connect(self._on_review)
        left_layout.addWidget(self.controls)

        # Right panel: Match list + Stats
        right_panel = QWidget()
        right_panel.setStyleSheet(PANEL_STYLE)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        right_panel.setLayout(right_layout)

        # Match list
        match_label = QLabel("📜 Match History")
        match_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(match_label)
        self.match_list = MatchListWidget()
        self.match_list.match_selected.connect(self._on_match_selected)
        right_layout.addWidget(self.match_list)

        # Stats panel
        stats_label = QLabel("📊 Statistics")
        stats_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(stats_label)
        self.stats_panel = StatsPanelWidget()
        right_layout.addWidget(self.stats_panel)

        # New match button
        new_match_btn = QPushButton("🎮 New Match")
        new_match_btn.setStyleSheet(BUTTON_STYLE)
        new_match_btn.clicked.connect(self._on_new_match)
        right_layout.addWidget(new_match_btn)

        right_layout.addStretch()

        # Splitter
        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _init_timer(self) -> None:
        """Initialize update timer."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_match_state)
        self.update_timer.start(2000)  # Update every 2 seconds

    def _show_login(self) -> None:
        """Show login dialog."""
        dialog = LoginDialog(self)
        
        # Create wrapper functions to pass dialog
        def on_login_signal(username: str, password: str):
            self._on_login(dialog, username, password)
        
        def on_register_signal(username: str, email: str, password: str):
            self._on_register(dialog, username, email, password)
        
        dialog.login_requested.connect(on_login_signal)
        dialog.register_requested.connect(on_register_signal)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Logged in")
            # Load initial data in background
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._load_initial_data())
                else:
                    loop.run_until_complete(self._load_initial_data())
            except RuntimeError:
                asyncio.run(self._load_initial_data())
        else:
            self.close()

    def _on_login(self, dialog: LoginDialog, username: str, password: str) -> None:
        """Handle login."""
        try:
            # Try to get event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use a different approach
                    import asyncio
                    import nest_asyncio
                    try:
                        nest_asyncio.apply()
                    except:
                        pass
                    result = loop.run_until_complete(self.api_client.login(username, password))
                else:
                    result = loop.run_until_complete(self.api_client.login(username, password))
            except RuntimeError:
                # No event loop, create one
                result = asyncio.run(self.api_client.login(username, password))
            
            QMessageBox.information(self, "Success", "Logged in successfully!")
            # Accept dialog to close it
            dialog.accept_dialog()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Login failed: {str(e)}")

    def _on_register(self, dialog: LoginDialog, username: str, email: str, password: str) -> None:
        """Handle register."""
        import threading
        
        def do_register():
            try:
                # Create new event loop in thread
                import asyncio
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(self.api_client.register(username, email, password))
                    # Use QTimer to update UI from main thread
                    QTimer.singleShot(0, lambda: self._on_register_success(dialog))
                except Exception as e:
                    error_msg = str(e)
                    QTimer.singleShot(0, lambda msg=error_msg: QMessageBox.critical(self, "Error", f"Registration failed: {msg}"))
                finally:
                    new_loop.close()
            except Exception as e:
                error_msg = str(e)
                QTimer.singleShot(0, lambda msg=error_msg: QMessageBox.critical(self, "Error", f"Registration failed: {msg}"))
        
        # Run in separate thread to avoid blocking UI
        thread = threading.Thread(target=do_register, daemon=True)
        thread.start()
    
    def _on_register_success(self, dialog: LoginDialog) -> None:
        """Handle successful registration."""
        QMessageBox.information(self, "Success", "Registered successfully!")
        # Accept dialog to close it
        dialog.accept_dialog()

    def _on_logout(self) -> None:
        """Handle logout."""
        self.api_client.clear_token()
        self.current_match_id = None
        self.board.clear_board()
        self.match_list.clear_matches()
        self._show_login()

    def _on_new_match(self) -> None:
        """Handle new match."""
        dialog = MatchDialog(self)
        dialog.match_created.connect(self._on_match_created)
        dialog.exec()

    def _on_match_created(self, match_type: str, level: int, board_size: int) -> None:
        """Handle match creation."""
        asyncio.create_task(self._create_match(match_type, level, board_size))

    async def _create_match(self, match_type: str, level: int, board_size: int) -> None:
        """Create match via API."""
        try:
            if match_type == "ai":
                result = await self.api_client.create_ai_match(level, board_size)
            else:
                result = await self.api_client.create_pvp_match(board_size)

            match_id = UUID(result["id"])
            self.current_match_id = match_id
            self.board.set_board_size(board_size)
            self.board.clear_board()
            self.move_number = 0

            # Determine player color (always Black for now)
            self.current_player_color = "B"

            self.statusBar().showMessage(f"Match created: {match_id}")
            await self._update_match_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create match: {str(e)}")

    def _on_board_click(self, x: int, y: int) -> None:
        """Handle board click."""
        if not self.current_match_id:
            return

        asyncio.create_task(self._submit_move(x, y))

    async def _submit_move(self, x: int, y: int) -> None:
        """Submit move."""
        if not self.current_match_id or not self.current_player_color:
            return

        try:
            self.move_number += 1
            result = await self.api_client.submit_move(
                self.current_match_id, x, y, self.move_number, self.current_player_color
            )

            # Update board
            self.board.set_stone(x, y, self.current_player_color)

            # Check for AI move
            if "ai_move" in result:
                ai_move = result["ai_move"]
                if ai_move and "x" in ai_move and "y" in ai_move:
                    ai_x = ai_move["x"]
                    ai_y = ai_move["y"]
                    ai_color = "W" if self.current_player_color == "B" else "B"
                    self.board.set_stone(ai_x, ai_y, ai_color)

            await self._update_match_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit move: {str(e)}")
            self.move_number -= 1

    def _on_pass(self) -> None:
        """Handle pass."""
        if not self.current_match_id or not self.current_player_color:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._submit_pass())
            else:
                loop.run_until_complete(self._submit_pass())
        except RuntimeError:
            asyncio.run(self._submit_pass())

    async def _submit_pass(self) -> None:
        """Submit pass."""
        try:
            self.move_number += 1
            await self.api_client.pass_turn(self.current_match_id, self.move_number, self.current_player_color)
            await self._update_match_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to pass: {str(e)}")
            self.move_number -= 1

    def _on_resign(self) -> None:
        """Handle resign."""
        if not self.current_match_id:
            return
        reply = QMessageBox.question(
            self, "Resign", "Are you sure you want to resign?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._submit_resign())
                else:
                    loop.run_until_complete(self._submit_resign())
            except RuntimeError:
                asyncio.run(self._submit_resign())

    async def _submit_resign(self) -> None:
        """Submit resign."""
        try:
            await self.api_client.resign_match(self.current_match_id)
            QMessageBox.information(self, "Resigned", "You have resigned the match.")
            self.current_match_id = None
            await self._load_initial_data()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resign: {str(e)}")

    def _on_hint(self) -> None:
        """Handle hint request."""
        if not self.current_match_id:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._request_hint())
            else:
                loop.run_until_complete(self._request_hint())
        except RuntimeError:
            asyncio.run(self._request_hint())

    async def _request_hint(self) -> None:
        """Request hint."""
        try:
            result = await self.api_client.request_hint(self.current_match_id, top_k=3)
            hints = result.get("hints", [])
            self.board.set_hints(hints)
            self.statusBar().showMessage(f"Hint: {len(hints)} suggestions")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get hint: {str(e)}")

    def _on_analysis(self) -> None:
        """Handle analysis request."""
        if not self.current_match_id:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._request_analysis())
            else:
                loop.run_until_complete(self._request_analysis())
        except RuntimeError:
            asyncio.run(self._request_analysis())

    async def _request_analysis(self) -> None:
        """Request analysis."""
        try:
            result = await self.api_client.request_analysis(self.current_match_id)
            analysis = result.get("analysis", {})
            win_prob = analysis.get("win_probability", 0.0)
            QMessageBox.information(
                self, "Analysis", f"Win Probability: {win_prob:.1%}\nEvaluation: {analysis.get('evaluation_score', 0)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get analysis: {str(e)}")

    def _on_review(self) -> None:
        """Handle review request."""
        if not self.current_match_id:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._request_review())
            else:
                loop.run_until_complete(self._request_review())
        except RuntimeError:
            asyncio.run(self._request_review())

    async def _request_review(self) -> None:
        """Request review."""
        try:
            result = await self.api_client.request_review(self.current_match_id)
            review = result.get("review", {})
            mistakes = review.get("mistakes", [])
            QMessageBox.information(self, "Review", f"Found {len(mistakes)} mistakes in the game.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get review: {str(e)}")

    def _on_match_selected(self, match_id: UUID) -> None:
        """Handle match selection."""
        self.current_match_id = match_id
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._load_match(match_id))
            else:
                loop.run_until_complete(self._load_match(match_id))
        except RuntimeError:
            asyncio.run(self._load_match(match_id))

    async def _load_match(self, match_id: UUID) -> None:
        """Load match state."""
        try:
            match = await self.api_client.get_match(match_id)
            board_size = match.get("board_size", 9)
            self.board.set_board_size(board_size)
            self.board.clear_board()

            # Load game state from replay
            replay = await self.api_client.get_replay(match_id)
            moves = replay.get("moves", [])

            for move in moves:
                pos = move.get("position")
                if pos and len(pos) == 2:
                    color = move.get("color", "B")
                    self.board.set_stone(pos[0], pos[1], color)

            self.statusBar().showMessage(f"Loaded match: {match_id}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load match: {str(e)}")

    @pyqtSlot()
    def _update_match_state(self) -> None:
        """Update match state (called by timer)."""
        if self.current_match_id:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._fetch_match_state())
                else:
                    loop.run_until_complete(self._fetch_match_state())
            except RuntimeError:
                pass  # Ignore if no event loop

    async def _fetch_match_state(self) -> None:
        """Fetch current match state."""
        if not self.current_match_id:
            return

        try:
            match = await self.api_client.get_match(self.current_match_id)
            self.current_match_state = match

            # Update board if needed
            game_state = match.get("game_state")
            if game_state:
                moves = game_state.get("moves", [])
                # Could update board here if needed
        except Exception:
            pass  # Ignore errors in background update

    async def _load_initial_data(self) -> None:
        """Load initial data (matches, stats)."""
        try:
            # Load match history
            matches = await self.api_client.get_match_history()
            self.match_list.clear_matches()
            for match in matches:
                match_id = UUID(match["id"])
                info = f"{match.get('result', 'Ongoing')} - {match.get('board_size', 9)}x{match.get('board_size', 9)}"
                self.match_list.add_match(match_id, info)

            # Load statistics
            stats = await self.api_client.get_my_statistics()
            self.stats_panel.update_stats(stats)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def closeEvent(self, event) -> None:
        """Handle window close."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.api_client.close())
            else:
                loop.run_until_complete(self.api_client.close())
        except RuntimeError:
            asyncio.run(self.api_client.close())
        event.accept()

