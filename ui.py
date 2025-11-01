import sys
import os
import json
import pathlib
import cv2
import numpy as np
import pyaudio
from rich import print, pretty

pretty.install()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QScrollArea, QFrame, QSizePolicy, QSplitter,
    QProgressBar, QStackedWidget
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont

from gtts import gTTS
import pygame
from faster_whisper import WhisperModel

from WelcomeWidget import WelcomeWidget
from QuizWidget import QuizWidget


class PracticeWidget(QWidget):
    check_work_requested = pyqtSignal = None
    ask_question_requested = pyqtSignal = None
    student_condition_updated = pyqtSignal = None

    def __init__(self):
        super().__init__()
        # define actual signals dynamically to avoid import-time PyQt complexities
        from PyQt6.QtCore import pyqtSignal
        self.check_work_requested = pyqtSignal()
        self.ask_question_requested = pyqtSignal()
        self.student_condition_updated = pyqtSignal(str)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(main_splitter)

        left_panel = QFrame(); left_layout = QVBoxLayout(left_panel)
        self.progress_bar = QProgressBar()

        symbol_font = QFont("Segoe UI", 16)

        self.question_display = QTextEdit("...")
        self.question_display.setFont(symbol_font)

        self.svg_widget = QSvgWidget()
        left_layout.addWidget(self.progress_bar); left_layout.addWidget(self.question_display, 1); left_layout.addWidget(self.svg_widget, 2)
        main_splitter.addWidget(left_panel)

        right_panel = QFrame(); right_layout = QVBoxLayout(right_panel)
        self.video_label = QLabel("Live Camera Feed"); self.video_label.setMinimumHeight(200)
        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True)
        chat_content = QWidget(); self.chat_layout = QVBoxLayout(chat_content); self.scroll_area.setWidget(chat_content)
        button_layout = QHBoxLayout(); self.check_work_btn = QPushButton("✅ Check My Work"); self.ask_question_btn = QPushButton("🤔 Ask a Question")
        button_layout.addWidget(self.check_work_btn); button_layout.addWidget(self.ask_question_btn)
        right_layout.addWidget(self.video_label); right_layout.addWidget(self.scroll_area); right_layout.addLayout(button_layout)
        main_splitter.addWidget(right_panel); main_splitter.setSizes([700, 500])

        self.check_work_btn.clicked.connect(lambda: self.check_work_requested.emit())
        self.ask_question_btn.clicked.connect(lambda: self.ask_question_requested.emit())

        self.stt_model = WhisperModel("tiny.en")

        self.cap = None
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self.update_webcam_feed)

        self.last_frame = None
        self.condition_timer = QTimer(self)
        self.condition_timer.timeout.connect(self.analyze_student_condition)
        self.condition_timer.start(2000)

    def activate_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.webcam_timer.start(30)
        print("Camera activated.")

    def update_webcam_feed(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
                    self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio
                ))

    def analyze_student_condition(self):
        if self.last_frame is None:
            return

        gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        condition = "confused" if blur < 30 else "attentive"
        self.student_condition_updated.emit(condition)

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

        try:
            with open("lessons\\quiz.json", "r", encoding="utf-8") as f:
                self.quiz_data = json.load(f)
        except FileNotFoundError:
            print("CRITICAL ERROR: quiz.json not found!")
            sys.exit()

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.welcome_widget = WelcomeWidget()
        self.quiz_widget = QuizWidget(self.quiz_data)
        self.practice_widget = PracticeWidget()

        self.stacked_widget.addWidget(self.welcome_widget)
        self.stacked_widget.addWidget(self.quiz_widget)
        self.stacked_widget.addWidget(self.practice_widget)

        # instantiate orchestrator here to avoid circular imports
        from orchestrator import TutorOrchestrator
        self.orchestrator = TutorOrchestrator(self)

        self.lesson_to_start = ""

        self.welcome_widget.start_quiz.connect(self.show_quiz)
        self.quiz_widget.quiz_reported.connect(self.show_report_and_speak)
        self.quiz_widget.proceed_to_learning.connect(self.start_tutor)

        self.show_welcome()

    def show_welcome(self):
        self.stacked_widget.setCurrentWidget(self.welcome_widget)

    def show_quiz(self):
        self.stacked_widget.setCurrentWidget(self.quiz_widget)

    def show_report_and_speak(self, score: int, weak_topic: str):
        print(f"Quiz finished with score: {score}. Weak topic: {weak_topic}")
        if score < 2:
            self.lesson_to_start = "lesson_remedial.json"
            tts_message = f"You did well! Let's work on {weak_topic} to get even better."
        else:
            self.lesson_to_start = "lesson_advanced.json"
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

    def update_svg(self, svg_path: str):
        self.practice_widget.svg_widget.load(svg_path)

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
