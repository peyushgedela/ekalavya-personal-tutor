from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QRadioButton, QButtonGroup, QFrame, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtSvgWidgets import QSvgWidget
import os


class QuizWidget(QWidget):
    """Displays MCQs, shows a report, and signals when to proceed."""
    quiz_reported = pyqtSignal(int, str)  # score, primary weak topic
    proceed_to_learning = pyqtSignal()

    def __init__(self, quiz_data):
        super().__init__()
        self.quiz_data = quiz_data
        self.current_question_index = 0
        self.score = 0
        self.strengths = []
        self.weaknesses = []

        self.setStyleSheet("background-color: #3C3C3C; color: white; font-size: 16px;")
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Quiz View ---
        self.quiz_frame = QFrame()
        quiz_layout = QVBoxLayout(self.quiz_frame)
        quiz_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # --- IMAGE HANDLING WIDGETS ---
        self.image_container = QStackedWidget()
        self.image_container.setMinimumHeight(300)
        self.image_container.setStyleSheet("background-color: #555;")

        self.svg_display = QSvgWidget()
        self.image_container.addWidget(self.svg_display)

        self.raster_display = QLabel()
        self.raster_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_container.addWidget(self.raster_display)
        # --- END IMAGE HANDLING ---

        self.question_label = QLabel()
        self.question_label.setWordWrap(True)
        self.question_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.options_group = QButtonGroup(self)
        self.options_layout = QVBoxLayout()
        
        self.submit_button = QPushButton("Next Question")
        self.submit_button.setFont(QFont("Segoe UI", 14))
        self.submit_button.clicked.connect(self.submit_answer)

        quiz_layout.addWidget(self.image_container, 2)
        quiz_layout.addWidget(self.question_label, 1)
        quiz_layout.addLayout(self.options_layout, 2)
        quiz_layout.addWidget(self.submit_button, 0, Qt.AlignmentFlag.AlignCenter)

        # --- Report View ---
        self.report_frame = QFrame()
        self.report_frame.hide() # Initially hidden
        report_layout = QVBoxLayout(self.report_frame)
        report_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.score_label = QLabel()
        self.score_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.strengths_label = QLabel()
        self.weaknesses_label = QLabel()
        
        self.proceed_button = QPushButton("Let's Start Learning!")
        self.proceed_button.setFont(QFont("Segoe UI", 16))
        self.proceed_button.clicked.connect(self.proceed_to_learning.emit)

        report_layout.addWidget(self.score_label)
        report_layout.addSpacing(20)
        report_layout.addWidget(self.strengths_label)
        report_layout.addWidget(self.weaknesses_label)
        report_layout.addSpacing(40)
        report_layout.addWidget(self.proceed_button)
        
        main_layout.addWidget(self.quiz_frame)
        main_layout.addWidget(self.report_frame)
        
        self.display_question()

    def display_question(self):
        for i in reversed(range(self.options_layout.count())): 
            widget = self.options_layout.itemAt(i).widget()
            if widget is not None: widget.setParent(None)

        question_data = self.quiz_data[self.current_question_index]
        self.question_label.setText(question_data["question"])

        # Image loading logic
        image_filename = question_data.get("svg_file")
        # If no filename provided, hide the image container to give more space to the question
        if not image_filename:
            self.image_container.setVisible(False)
        else:
            self.image_container.setVisible(True)
            image_path = os.path.join("assets", image_filename)

            if not os.path.exists(image_path):
                print(f"Error: [QuizWidget] Cannot find image {image_path}")
                self.raster_display.setText("Image not found")
                self.raster_display.setPixmap(QPixmap())
                self.image_container.setCurrentWidget(self.raster_display)

            elif image_filename.lower().endswith('.svg'):
                # SVG viewer handles its own scaling
                try:
                    self.svg_display.load(image_path)
                except Exception:
                    self.raster_display.setText("Failed to load SVG")
                    self.raster_display.setPixmap(QPixmap())
                    self.image_container.setCurrentWidget(self.raster_display)
                else:
                    self.image_container.setCurrentWidget(self.svg_display)

            elif image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                pixmap = QPixmap(image_path)
                if pixmap.isNull():
                    self.raster_display.setText("Image failed to load")
                    self.raster_display.setPixmap(QPixmap())
                else:
                    scaled_pixmap = pixmap.scaled(
                        self.raster_display.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.raster_display.setPixmap(scaled_pixmap)
                self.image_container.setCurrentWidget(self.raster_display)

            else:
                self.raster_display.setText("Unsupported image format")
                self.raster_display.setPixmap(QPixmap())
                self.image_container.setCurrentWidget(self.raster_display)

        for option in question_data["options"]:
            radio_button = QRadioButton(option)
            self.options_layout.addWidget(radio_button)
            self.options_group.addButton(radio_button)

        if self.current_question_index == len(self.quiz_data) - 1:
            self.submit_button.setText("Finish Quiz")

    def submit_answer(self):
        selected_button = self.options_group.checkedButton()
        if not selected_button: return

        question_data = self.quiz_data[self.current_question_index]
        topic = question_data.get("topic", "Unnamed Topic")
        if selected_button.text() == question_data["answer"]:
            self.score += 1
            self.strengths.append(topic)
        else:
            self.weaknesses.append(topic)

        self.current_question_index += 1
        if self.current_question_index < len(self.quiz_data):
            self.display_question()
        else:
            self.show_report()

    def show_report(self):
        self.quiz_frame.hide()
        self.report_frame.show()

        total_questions = len(self.quiz_data)
        self.score_label.setText(f"Quiz Complete! Your Score: {self.score}/{total_questions}")
        
        strengths_text = "<b>Strengths:</b> " + (", ".join(self.strengths) if self.strengths else "None")
        self.strengths_label.setText(strengths_text)

        weaknesses_text = "<b>Areas to Improve:</b> " + (", ".join(self.weaknesses) if self.weaknesses else "None! Great job!")
        self.weaknesses_label.setText(weaknesses_text)

        primary_weak_topic = self.weaknesses[0] if self.weaknesses else "all the concepts"
        self.quiz_reported.emit(self.score, primary_weak_topic)

