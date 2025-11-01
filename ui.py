import sys
import os
import json
import pathlib
import cv2
import numpy as np
import pyaudio
# mediapipe can fail to import on Windows if the native DLLs / redistributables
# aren't present or if there's an architecture mismatch. Import in a try/except
# so the app can still run (head-pose features will be disabled).
try:
    import mediapipe as mp
except Exception as _err:  # keep broad to capture DLL import failures
    mp = None
    print(f"Warning: mediapipe import failed: {_err}")
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
        self.svg_widget = QSvgWidget()
        left_layout.addWidget(self.progress_bar); left_layout.addWidget(self.question_display, 1); left_layout.addWidget(self.svg_widget, 2)
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
        
        # --- Initialize MediaPipe (if available) ---
        if mp is not None:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                self.status_label.setText("Status: Ready (head-pose enabled)")
            except Exception as e:
                # If MediaPipe failed to initialize, disable pose features
                print(f"Warning: mediapipe initialization failed: {e}")
                self.mp_face_mesh = None
                self.face_mesh = None
                self.status_label.setText("Status: Mediapipe init failed — head-pose disabled")
        else:
            # mediapipe import failed earlier; disable pose features
            self.mp_face_mesh = None
            self.face_mesh = None
            self.status_label.setText("Status: Mediapipe unavailable — head-pose disabled")
        
        # --- Initialize Webcam ---
        self.cap = None
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self.update_frame_and_pose) # Renamed this function
        
        # Remove the old condition_timer
        self.last_frame = None

    def activate_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.webcam_timer.start(30) # 30ms timer for ~33 FPS
        print("Camera and Head Pose analysis activated.")
    
    def update_frame_and_pose(self):
        """This function now does both: updates the video feed AND analyzes pose."""
        if not (self.cap and self.cap.isOpened()):
            return

        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Flip the frame for a "selfie" view
        frame = cv2.flip(frame, 1)
        self.last_frame = frame # Store the flipped frame

        # Analyze the pose first
        self.analyze_head_pose(frame)

        # Convert for Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio
        ))
    
    def analyze_head_pose(self, frame):
        """
        Analyzes the frame to find head pose and updates the status.
        Replaces the old analyze_student_condition.
        """
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Performance optimization
        results = self.face_mesh.process(frame_rgb)
        
        status_text = "Not Detected"
        status_color = "#AAAAAA" # Grey

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # --- Head Pose Logic ---
            # This is a simplified 3D-to-2D calculation (SolvePnP)
            # We get key points from MediaPipe's 468 landmarks
            h, w, _ = frame.shape
            face_2d = []
            face_3d = []
            
            # Key points: Nose, Chin, Left/Right Eye, Left/Right Mouth
            key_points_indices = [1, 152, 263, 33, 287, 57] 
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in key_points_indices:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_2d.append([x, y])
            
            # These are generic 3D model coordinates
            face_3d = [
                [0.0, 0.0, 0.0],    # Nose tip (1)
                [0.0, -330.0, -65.0], # Chin (152)
                [-225.0, 170.0, -135.0], # Left eye left corner (263)
                [225.0, 170.0, -135.0],  # Right eye right corner (33)
                [-150.0, -150.0, -125.0], # Left Mouth corner (287)
                [150.0, -150.0, -125.0]   # Right mouth corner (57)
            ]
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]])
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            if success:
                # Get rotation matrix
                rmat, _ = cv2.Rodrigues(rot_vec)
                
                # Get pitch, yaw, roll (in degrees)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch, yaw, roll = angles[0], angles[1], angles[2]

                # --- Define States based on Head Pose ---
                if pitch > 20: # Looking down
                    status_text = "Desk Focus"
                    status_color = "#2ECC71" # Green
                elif abs(yaw) > 30 or pitch < -15: # Looking side-to-side or up
                    status_text = "Away"
                    status_color = "#F1C40F" # Yellow
                else: # Looking forward
                    status_text = "Screen Focus"
                    status_color = "#50A0FF" # Blue
            else:
                status_text = "Error: PnP Failed"
        
        # --- Update the UI and Emit the Signal ---
        self.status_label.setText(f"Status: {status_text}")
        self.status_label.setStyleSheet(f"color: {status_color}; font-weight: bold; font-size: 14px;")
        
        # Send this new, richer state to the orchestrator
        self.student_condition_updated.emit(status_text)

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
