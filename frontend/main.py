"""GoGame Frontend - Entry Point."""

import asyncio
import sys

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("GoGame")

    # Create main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

