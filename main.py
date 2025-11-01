"""Entry point for Ekalavya AI Tutor — delegates implementation to modules.

This file is intentionally small so multiple developers can work on
`ui.py`, `orchestrator.py`, `analysis.py`, and `utils.py` independently.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

# Import the MainWindow from the new ui module
from ui import MainWindow


def main():
    app = QApplication(sys.argv)
    default_font = QFont("Segoe UI", 12)
    app.setFont(default_font)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

