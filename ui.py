import sys
import os
import json
import pathlib
import cv2
import numpy as np
import pyaudio
import base64
import time
from datetime import datetime
from groq import Groq
from rich import print, pretty

pretty.install()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QScrollArea, QFrame, QSizePolicy, QSplitter,
    QProgressBar, QStackedWidget
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont

from gtts import gTTS
import pygame
from faster_whisper import WhisperModel

from WelcomeWidget import WelcomeWidget
from QuizWidget import QuizWidget
from TopicsWidget import TopicsWidget


class PracticeWidget(QWidget):
    check_work_requested = pyqtSignal()
    ask_question_requested = pyqtSignal()
    student_condition_updated = pyqtSignal(str) # This signal will now send "Screen Focus", "Desk Focus", etc.

    def __init__(self):
        super().__init__()
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(main_splitter)
        
        # --- Left Panel (No changes) ---
        left_panel = QFrame(); left_layout = QVBoxLayout(left_panel)
        self.progress_bar = QProgressBar(); 
        self.question_display = QTextEdit("..."); 
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.progress_bar); left_layout.addWidget(self.question_display, 1); left_layout.addWidget(self.image_label, 2)
        main_splitter.addWidget(left_panel)
        
        # --- Right Panel (MODIFIED) ---
        right_panel = QFrame(); right_layout = QVBoxLayout(right_panel)
        
        # 1. Webcam Feed
        self.video_label = QLabel("Initializing Camera..."); 
        self.video_label.setMinimumHeight(200)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #222; border-radius: 5px;")

        # 2. NEW: Status Label (for our new detection)
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 3. Chat Area
        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True)
        chat_content = QWidget(); self.chat_layout = QVBoxLayout(chat_content); self.scroll_area.setWidget(chat_content)
        
        # 4. Buttons
        button_layout = QHBoxLayout(); self.check_work_btn = QPushButton("✅ Check My Work"); self.ask_question_btn = QPushButton("🤔 Ask a Question")
        button_layout.addWidget(self.check_work_btn); button_layout.addWidget(self.ask_question_btn)
        
        # Add all widgets to the right layout
        right_layout.addWidget(self.video_label)
        right_layout.addWidget(self.status_label) # Add the new label
        right_layout.addWidget(self.scroll_area); 
        right_layout.addLayout(button_layout)
        
        main_splitter.addWidget(right_panel); main_splitter.setSizes([700, 500])

        self.check_work_btn.clicked.connect(self.check_work_requested.emit)
        self.ask_question_btn.clicked.connect(self.ask_question_requested.emit)
        
        # --- Initialize ASR (no change) ---
        self.stt_model = WhisperModel("tiny.en")
        
        # --- Initialize Webcam Timer ---
        self.cap = None
        self.webcam_timer = QTimer(self)  # For frame updates
        self.webcam_timer.timeout.connect(self.update_frame)
        self.last_frame = None
        
        # Commented out engagement analysis for now
        # self.client = Groq()
        # self.analysis_timer = QTimer(self)  # For periodic analysis
        # self.analysis_timer.timeout.connect(self.analyze_student_state)
        # self.last_analysis_time = 0
        
        self.status_label.setText("Status: Ready (periodic analysis enabled)")

    def activate_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.webcam_timer.start(30)  # 30ms timer for ~33 FPS
        print("Camera activated.")
    
    def update_frame(self):
        """Updates the video feed display."""
        if not (self.cap and self.cap.isOpened()):
            return

        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Flip the frame for a "selfie" view
        frame = cv2.flip(frame, 1)
        self.last_frame = frame  # Store the flipped frame

        # Convert for Qt display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio
        ))
    
    def analyze_student_state(self):
        """
        Temporarily disabled - will be re-implemented later.
        """
        # Analysis functionality is commented out
        pass


        


    def start_listening(self):
        self.ask_question_btn.setText("Listening..."); self.ask_question_btn.setEnabled(False)
        QApplication.processEvents()
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = [stream.read(1024) for _ in range(0, int(16000 / 1024 * 5))]
        stream.stop_stream(); stream.close(); p.terminate()
        audio_np = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.stt_model.transcribe(audio_np, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
        self.ask_question_btn.setEnabled(True)
        return text

    def add_message(self, sender, message):
        bubble = QTextEdit(message); bubble.setReadOnly(True)
        align = Qt.AlignmentFlag.AlignLeft if sender == "Assistant" else Qt.AlignmentFlag.AlignRight
        style = ("background-color: #E5E5EA; color: black;" if sender == "Assistant" else "background-color: #00519E; color: white;")

        symbol_font = QFont("Segoe UI", 14)
        bubble.setFont(symbol_font)

        bubble.setStyleSheet(f"{style} border-radius: 10px; padding: 10px;")
        self.chat_layout.addWidget(bubble, alignment=align)
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum()))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ekalavya AI Tutor")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F6FA;
            }
        """)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.welcome_widget = WelcomeWidget()
        self.topics_widget = TopicsWidget()
        self.quiz_widget = None  # Will be initialized when topic is selected
        self.practice_widget = None  # Will be initialized when needed

        self.stacked_widget.addWidget(self.welcome_widget)
        self.stacked_widget.addWidget(self.topics_widget)

        # instantiate orchestrator here to avoid circular imports
        from orchestrator import TutorOrchestrator
        self.orchestrator = TutorOrchestrator(self)

        self.current_topic_path = ""
        self.lesson_to_start = ""

        # Connect signals
        self.welcome_widget.start_quiz.connect(self.show_topics)
        self.topics_widget.topic_selected.connect(self.topic_selected)

        self.show_welcome()

    def topic_selected(self, topic_path: str):
        """Called when a topic is selected from the topics screen"""
        self.current_topic_path = topic_path
        try:
            quiz_path = os.path.join(topic_path, "quiz.json")
            with open(quiz_path, "r", encoding="utf-8") as f:
                quiz_data = json.load(f)
        except FileNotFoundError:
            print(f"CRITICAL ERROR: quiz.json not found in {topic_path}!")
            return

        # Initialize quiz and practice widgets for this topic
        self.quiz_widget = QuizWidget(quiz_data)
        self.practice_widget = PracticeWidget()
        
        # Add widgets to stacked widget if not already added
        if self.stacked_widget.indexOf(self.quiz_widget) == -1:
            self.stacked_widget.addWidget(self.quiz_widget)
        if self.stacked_widget.indexOf(self.practice_widget) == -1:
            self.stacked_widget.addWidget(self.practice_widget)

        # Connect signals for the new widgets
        self.quiz_widget.quiz_reported.connect(self.show_report_and_speak)
        self.quiz_widget.proceed_to_learning.connect(self.start_tutor)
        
        # Connect practice widget signals in the orchestrator
        self.orchestrator.connect_practice_widget_signals()

        self.show_quiz()

    def show_welcome(self):
        self.stacked_widget.setCurrentWidget(self.welcome_widget)

    def show_topics(self):
        self.stacked_widget.setCurrentWidget(self.topics_widget)

    def show_quiz(self):
        self.stacked_widget.setCurrentWidget(self.quiz_widget)

    def show_report_and_speak(self, score: int, weak_topic: str):
        print(f"Quiz finished with score: {score}. Weak topic: {weak_topic}")
        if score < 3:
            self.lesson_to_start = os.path.join(self.current_topic_path, "lesson_remedial.json")
            tts_message = f"You did well! Let's work on {weak_topic} to get even better."
        else:
            self.lesson_to_start = os.path.join(self.current_topic_path, "lesson_advanced.json")
            tts_message = f"Excellent work! Let's review {weak_topic} to perfect your skills."

        self.orchestrator.speak(tts_message)

    def start_tutor(self):
        print(f"Proceeding to lesson: {self.lesson_to_start}")
        self.orchestrator.stop_tts()
        self.practice_widget.activate_camera()
        self.stacked_widget.setCurrentWidget(self.practice_widget)
        self.orchestrator.start(self.lesson_to_start)

    def add_chat_message(self, sender: str, message: str):
        self.practice_widget.add_message(sender, message)

    def update_question(self, text: str):
        self.practice_widget.question_display.setText(text)

    def update_image(self, image_path: str):
        """Update the image in the practice widget. Supports SVG, PNG, and JPEG formats."""
        if image_path.lower().endswith('.svg'):
            svg_widget = QSvgWidget()
            svg_widget.load(image_path)
            svg_widget.setMinimumSize(400, 400)
            self.practice_widget.image_label.setWidget(svg_widget)
        else:
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.practice_widget.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.practice_widget.image_label.setPixmap(scaled_pixmap)

    def update_progress(self, completed_steps: int, total_steps: int):
        self.practice_widget.progress_bar.setMaximum(max(1, total_steps))
        self.practice_widget.progress_bar.setValue(min(completed_steps, total_steps))

    def set_button_state(self, mode: str, enabled: bool = True):
        if mode == "instruction":
            self.practice_widget.check_work_btn.setEnabled(False)
            self.practice_widget.ask_question_btn.setEnabled(True if enabled else False)
        elif mode == "practice":
            self.practice_widget.check_work_btn.setEnabled(True if enabled else False)
            self.practice_widget.ask_question_btn.setEnabled(True if enabled else False)
        else:
            self.practice_widget.check_work_btn.setEnabled(enabled)
            self.practice_widget.ask_question_btn.setEnabled(enabled)

    def listen_for_question(self) -> str:
        return self.practice_widget.start_listening()
