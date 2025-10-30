from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QRadioButton, QButtonGroup, QFrame
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class QuizWidget(QWidget):
    """Displays MCQs, shows a report, and signals when to proceed."""
    # Signal now emits the score and the main topic to work on
    quiz_reported = pyqtSignal(int, str) 
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
        
        self.question_label = QLabel()
        self.question_label.setWordWrap(True)
        self.question_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        
        self.options_group = QButtonGroup(self)
        self.options_layout = QVBoxLayout()
        
        self.submit_button = QPushButton("Next Question")
        self.submit_button.setFont(QFont("Segoe UI", 14))
        self.submit_button.clicked.connect(self.submit_answer)

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

        # Determine the primary topic to work on for the TTS message
        primary_weak_topic = self.weaknesses[0] if self.weaknesses else "all the concepts"
        
        # Emit the signal so MainWindow can trigger the TTS
        self.quiz_reported.emit(self.score, primary_weak_topic)

