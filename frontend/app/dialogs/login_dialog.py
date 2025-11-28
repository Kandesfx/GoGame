"""Login/Register dialog."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QPushButton, QTabWidget, QWidget

from ..styles import BUTTON_STYLE, DIALOG_STYLE, FORM_STYLE


class LoginDialog(QDialog):
    """Dialog cho login/register."""

    login_requested = pyqtSignal(str, str)  # username_or_email, password
    register_requested = pyqtSignal(str, str, str)  # username, email, password

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login / Register")
        self.setMinimumWidth(350)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QFormLayout()
        layout.setSpacing(15)
        self.setLayout(layout)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Login tab
        login_widget = QWidget()
        login_layout = QFormLayout()
        login_widget.setLayout(login_layout)

        self.login_username = QLineEdit()
        self.login_password = QLineEdit()
        self.login_password.setEchoMode(QLineEdit.EchoMode.Password)
        login_layout.addRow("Username/Email:", self.login_username)
        login_layout.addRow("Password:", self.login_password)
        login_widget.setStyleSheet(FORM_STYLE)

        login_btn = QPushButton("🔐 Login")
        login_btn.setStyleSheet(BUTTON_STYLE)
        login_btn.clicked.connect(self._on_login)
        login_layout.addRow(login_btn)

        tabs.addTab(login_widget, "Login")

        # Register tab
        register_widget = QWidget()
        register_layout = QFormLayout()
        register_widget.setLayout(register_layout)

        self.register_username = QLineEdit()
        self.register_email = QLineEdit()
        self.register_password = QLineEdit()
        self.register_password.setEchoMode(QLineEdit.EchoMode.Password)
        register_layout.addRow("Username:", self.register_username)
        register_layout.addRow("Email:", self.register_email)
        register_layout.addRow("Password:", self.register_password)
        register_widget.setStyleSheet(FORM_STYLE)

        register_btn = QPushButton("✨ Register")
        register_btn.setStyleSheet(BUTTON_STYLE)
        register_btn.clicked.connect(self._on_register)
        register_layout.addRow(register_btn)

        tabs.addTab(register_widget, "Register")

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_login(self) -> None:
        """Handle login button."""
        username = self.login_username.text()
        password = self.login_password.text()
        if username and password:
            self.login_requested.emit(username, password)
            # Note: Dialog will be accepted in main_window after successful login

    def _on_register(self) -> None:
        """Handle register button."""
        username = self.register_username.text()
        email = self.register_email.text()
        password = self.register_password.text()
        if username and email and password:
            self.register_requested.emit(username, email, password)
            # Note: Dialog will be accepted in main_window after successful registration
    
    def accept_dialog(self) -> None:
        """Accept dialog (called after successful login/register)."""
        self.accept()

