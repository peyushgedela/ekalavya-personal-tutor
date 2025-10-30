from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class WelcomeWidget(QWidget):
    """The first screen the user sees. Greets the user and starts the quiz."""
    start_quiz = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("color: white;")

        title = QLabel("Welcome to Ekalavya")
        title.setFont(QFont("Segoe UI", 40, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Your Personal AI Learning Companion")
        subtitle.setFont(QFont("Segoe UI", 18))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_button = QPushButton("Start Learning")
        self.start_button.setFont(QFont("Segoe UI", 16))
        self.start_button.setMinimumHeight(60)
        self.start_button.setStyleSheet("""
            QPushButton { 
                background-color: #2ECC71; 
                color: white; 
                border-radius: 10px; 
                padding: 10px;
            }
            QPushButton:hover { background-color: #27ae60; }
        """)
        self.start_button.clicked.connect(self.start_quiz.emit)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(50)
        layout.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignCenter)
