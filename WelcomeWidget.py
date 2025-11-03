from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class WelcomeWidget(QWidget):
    """The first screen the user sees. Greets the user and starts the quiz."""
    start_quiz = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                color: #2C3E50;
            }
        """)

        # Create a card-like container
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 20px;
                padding: 40px;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(30)

        # Welcome header with accent color
        header = QFrame()
        header.setStyleSheet("background-color: #00519E; border-radius: 15px; padding: 20px;")
        header_layout = QVBoxLayout(header)

        title = QLabel("Welcome to Ekalavya")
        title.setFont(QFont("Segoe UI", 40, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white;")

        subtitle = QLabel("Your Personal AI Learning Companion")
        subtitle.setFont(QFont("Segoe UI", 18))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #E5E5EA;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        # Description text
        description = QLabel(
            "Ready to explore the fascinating world of Mathematics?\n"
            "Start your journey with personalized learning experience."
        )
        description.setFont(QFont("Segoe UI", 14))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("color: #7F8C8D; padding: 20px;")
        description.setWordWrap(True)

        # Styled start button
        self.start_button = QPushButton("Start Learning")
        self.start_button.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.start_button.setMinimumHeight(60)
        self.start_button.setMinimumWidth(200)
        self.start_button.setStyleSheet("""
            QPushButton { 
                background-color: #2ECC71; 
                color: white; 
                border-radius: 30px; 
                padding: 15px 30px;
            }
            QPushButton:hover { 
                background-color: #27ae60;
                margin-top: -2px;
                margin-bottom: 2px;
                border-bottom: 4px solid rgba(46, 204, 113, 0.2);
            }
        """)
        self.start_button.clicked.connect(self.start_quiz.emit)

        # Add everything to the card layout
        card_layout.addWidget(header)
        card_layout.addWidget(description)
        card_layout.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignCenter)

        # Add card to main layout with some margin
        layout.addWidget(card)
        layout.setContentsMargins(40, 40, 40, 40)
