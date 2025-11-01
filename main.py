# main.py
# Ekalavya AI Tutor - Simplified Reactive Model

import sys
import os
import json
import time
import threading
import pathlib
import cv2
import numpy as np
import pyaudio
import base64
import re
from rich import print, pretty

pretty.install()

# --- Core Dependencies ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QScrollArea, QFrame, QSizePolicy, QSplitter,
    QProgressBar, QStackedWidget
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage

# --- External Libraries ---
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
import pygame
from faster_whisper import WhisperModel

# --- Internal Libraries ---
from WelcomeWidget import WelcomeWidget
from QuizWidget import QuizWidget
from OverheadCamera import OverheadCamera

# ==============================================================================
#  1. SENSING & ANALYSIS LOGIC
# ==============================================================================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_from_llm_output(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: return {"error": "Invalid JSON from analysis model."}
    return {"error": "No JSON object found in analysis response."}

class AnalysisWorker(QObject):
    analysis_complete = pyqtSignal(dict)
    def __init__(self, overhead_camera_instance):
        super().__init__()
        self.camera = overhead_camera_instance
    def run(self):
        result = self.camera.capture_and_analyze(encode_image, extract_json_from_llm_output)
        self.analysis_complete.emit(result)

# ==============================================================================
#  2. THE USER INTERFACE
# ==============================================================================

class PracticeWidget(QWidget):
    check_work_requested = pyqtSignal()
    ask_question_requested = pyqtSignal()
    student_condition_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(main_splitter)
        
        left_panel = QFrame(); left_layout = QVBoxLayout(left_panel)
        self.progress_bar = QProgressBar(); self.question_display = QTextEdit("..."); self.svg_widget = QSvgWidget()
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

        self.check_work_btn.clicked.connect(self.check_work_requested.emit)
        self.ask_question_btn.clicked.connect(self.ask_question_requested.emit)
        
        self.stt_model = WhisperModel("tiny.en")
        
        self.cap = None
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self.update_webcam_feed)

        # NEW: Separate timer for less frequent condition analysis
        self.last_frame = None
        self.condition_timer = QTimer(self)
        self.condition_timer.timeout.connect(self.analyze_student_condition)
        self.condition_timer.start(2000) # Analyze every 2 seconds
    def activate_camera(self):
        """NEW METHOD: Called by MainWindow to turn on the camera."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.webcam_timer.start(30)
        print("Camera activated.")
    
    def update_webcam_feed(self):
        if self.cap and self.cap.isOpened(): # NEW: Check if camera is active
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
        """Analyzes the last captured frame and emits the student's condition."""
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
        bubble.setStyleSheet(f"{style} border-radius: 10px; padding: 10px; font-size: 14px;")
        self.chat_layout.addWidget(bubble, alignment=align)
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum()))

class MainWindow(QMainWindow):
    """The main window that controls the application flow."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ekalavya AI Tutor")
        self.setGeometry(100, 100, 1200, 800)

        # --- Load Quiz Data ---
        try:
            with open("lessons\quiz.json", "r") as f:
                self.quiz_data = json.load(f)
        except FileNotFoundError:
            print("CRITICAL ERROR: quiz.json not found!")
            # You can add a popup error message here for the user
            sys.exit()

        # --- Central Widget to Switch Screens ---
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # --- Create Widgets from your new modules ---
        self.welcome_widget = WelcomeWidget()
        self.quiz_widget = QuizWidget(self.quiz_data)
        self.practice_widget = PracticeWidget() # The main tutor UI

        # --- Add Widgets to Stack ---
        self.stacked_widget.addWidget(self.welcome_widget)
        self.stacked_widget.addWidget(self.quiz_widget)
        self.stacked_widget.addWidget(self.practice_widget)

        # --- Initialize Orchestrator ---
        self.orchestrator = TutorOrchestrator(self)
        
        self.lesson_to_start = "" # NEW: To store the chosen lesson plan

        # --- Connect Signals for the new flow ---
        self.welcome_widget.start_quiz.connect(self.show_quiz)
        self.quiz_widget.quiz_reported.connect(self.show_report_and_speak) # UPDATED
        self.quiz_widget.proceed_to_learning.connect(self.start_tutor) # NEW

        self.show_welcome()

    def show_welcome(self):
        self.stacked_widget.setCurrentWidget(self.welcome_widget)

    def show_quiz(self):
        self.stacked_widget.setCurrentWidget(self.quiz_widget)

    def show_report_and_speak(self, score: int, weak_topic: str):
        """Triggered when the quiz finishes to play the TTS message."""
        print(f"Quiz finished with score: {score}. Weak topic: {weak_topic}")
        
        # Determine which lesson to load later
        if score < 2:
            self.lesson_to_start = "lesson_remedial.json"
            tts_message = f"You did well! Let's work on {weak_topic} to get even better."
        else:
            self.lesson_to_start = "lesson_advanced.json"
            tts_message = f"Excellent work! Let's review {weak_topic} to perfect your skills."

        self.orchestrator.speak(tts_message) # Use the new TTS-only method

    def start_tutor(self):
        """Triggered by the 'Proceed' button. Stops audio, starts camera, and begins the lesson."""
        print(f"Proceeding to lesson: {self.lesson_to_start}")
        
        self.orchestrator.stop_tts()
        self.practice_widget.activate_camera() # Turn on the camera NOW
        
        self.stacked_widget.setCurrentWidget(self.practice_widget)
        self.orchestrator.start(self.lesson_to_start)

    # --- UI wrapper methods expected by TutorOrchestrator ---
    def add_chat_message(self, sender: str, message: str):
        self.practice_widget.add_message(sender, message)

    def update_question(self, text: str):
        self.practice_widget.question_display.setText(text)

    def update_svg(self, svg_path: str):
        # QSvgWidget.load accepts a file path (str) or QByteArray
        self.practice_widget.svg_widget.load(svg_path)

    def update_progress(self, completed_steps: int, total_steps: int):
        self.practice_widget.progress_bar.setMaximum(max(1, total_steps))
        self.practice_widget.progress_bar.setValue(min(completed_steps, total_steps))

    def set_button_state(self, mode: str, enabled: bool = True):
        # Modes: "instruction" or "practice"
        if mode == "instruction":
            self.practice_widget.check_work_btn.setEnabled(False)
            self.practice_widget.ask_question_btn.setEnabled(True if enabled else False)
        elif mode == "practice":
            self.practice_widget.check_work_btn.setEnabled(True if enabled else False)
            self.practice_widget.ask_question_btn.setEnabled(True if enabled else False)
        else:
            # Fallback: toggle both
            self.practice_widget.check_work_btn.setEnabled(enabled)
            self.practice_widget.ask_question_btn.setEnabled(enabled)

    def listen_for_question(self) -> str:
        return self.practice_widget.start_listening()

# ==============================================================================
#  3. THE BRAIN (TutorOrchestrator
# ==============================================================================

class TutorOrchestrator:
    def __init__(self, main_window: MainWindow):
        self.ui = main_window
        load_dotenv()
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        pygame.mixer.init()

        self.base_dir = pathlib.Path(__file__).parent
        try:
            with open(self.base_dir / "prompts" / "hint_prompt.txt", "r", encoding="utf-8") as f:
                self.hint_prompt_template = f.read()
            with open(self.base_dir / "prompts" / "validation_prompt.txt", "r", encoding="utf-8") as f:
                self.validation_prompt_template = f.read()
            with open(self.base_dir / "prompts" / "ocr_prompt.txt", "r", encoding="utf-8") as f:
                self.ocr_prompt_template = f.read()
            print("Prompt templates loaded successfully.")
        except FileNotFoundError as e:
            print(f"CRITICAL ERROR: Could not find prompt files. {e}")
            self.hint_prompt_template = "Error: Hint prompt file not found."
            self.validation_prompt_template = "Error: Validation prompt file not found."
            self.ocr_prompt_template = "Error: OCR prompt file not found."
        
        ip_camera_url = os.environ.get("IP_CAMERA_URL", "http://192.168.0.101:8080/video")
        self.overhead_camera = OverheadCamera(ip_url=ip_camera_url, groq_client=self.groq_client, ocr_prompt=self.ocr_prompt_template)
        
        self.state = "IDLE"
        self.current_lesson = None
        self.current_step_index = 0
        self.attempt_counter = 0
        self.student_condition = "attentive"
        
        self.ui.practice_widget.check_work_requested.connect(self.on_check_work)
        self.ui.practice_widget.ask_question_requested.connect(self.on_ask_question)
        self.ui.practice_widget.student_condition_updated.connect(self.on_student_condition_update) # Connect to the new signal

    def on_student_condition_update(self, condition: str):
        """Receives the condition from the UI and updates the state."""
        if self.student_condition != condition:
            self.student_condition = condition
            print(f"Student condition updated to: {self.student_condition}")

    def start(self, filename: str | None = None):
        if not filename:
            filename = "lesson_remedial.json"
        self.load_lesson(filename)
    
    def load_lesson(self, filename):
        try:
            with open(self.base_dir / "lessons" / filename, 'r') as f: self.current_lesson = json.load(f)
            self.current_step_index = 0; self.process_current_step()
        except FileNotFoundError: self.ui.add_chat_message("Assistant", f"Error: Lesson file '{filename}' not found.")

    def process_current_step(self):
        if self.current_step_index >= len(self.current_lesson):
            self.speak_and_show("Great job! You've completed all the steps.", "Assistant"); self.ui.set_button_state("practice", enabled=False); return
        step = self.current_lesson[self.current_step_index]
        self.ui.update_question(step['question_text'])
        
        svg_path = str(self.base_dir / "assets" / step['svg_file'])
        if os.path.exists(svg_path):
            self.ui.update_svg(svg_path)
        else:
            print(f"Warning: SVG file not found at {svg_path}")
            self.ui.add_chat_message("Assistant", f"(Diagram '{step['svg_file']}' is missing)")

        validation_steps = [s for s in self.current_lesson if s['type'] == 'validation']
        completed_steps = len([s for s in self.current_lesson[:self.current_step_index] if s['type'] == 'validation'])
        self.ui.update_progress(completed_steps, len(validation_steps))
        
        if step['type'] == 'instruction':
            self.state = "INSTRUCTION"
            self.ui.set_button_state("instruction")
            self.speak_and_show(step['feedback_text'], "Assistant")
            
            def wait_and_advance():
                if pygame.mixer.music.get_busy():
                    QTimer.singleShot(500, wait_and_advance)
                else:
                    self.advance_step()
            wait_and_advance()
        else:
            self.state = "PRACTICE"; self.ui.set_button_state("practice")
            if self.attempt_counter == 0: self.speak_and_show("Okay, your turn...", "Assistant")

    def advance_step(self):
        if self.state == "INSTRUCTION": self.current_step_index += 1; self.process_current_step()
        elif self.state == "PRACTICE": self.attempt_counter = 0; self.current_step_index += 1; self.process_current_step()

    def on_check_work(self):
        self.ui.set_button_state("practice", enabled=False); self.speak_and_show("Okay, let me check...", "Assistant")
        self.worker = AnalysisWorker(self.overhead_camera); self.thread = threading.Thread(target=self.worker.run)
        self.worker.analysis_complete.connect(self.on_analysis_complete); self.thread.start()

    def on_analysis_complete(self, result):
        if "error" in result: self.speak_and_show(f"Error: {result['error']}", "Assistant"); self.ui.set_button_state("practice", enabled=True); return
        analysis = result.get("analysis", {}); ocr_text = "\n".join(analysis.get("lines", []))
        if not ocr_text.strip(): self.speak_and_show("I didn't see any writing.", "Assistant"); self.ui.set_button_state("practice", enabled=True); return
        self.ui.add_chat_message("Assistant", f"I see: \"{ocr_text}\""); step = self.current_lesson[self.current_step_index]
        is_complete = self.validate_with_llm(ocr_text, step)
        if is_complete: self.speak_and_show("Perfect!", "Assistant"); self.advance_step()
        else: self.attempt_counter += 1; hint = self.generate_hint_with_llm(ocr_text, step); self.speak_and_show(hint, "Assistant"); self.ui.set_button_state("practice", enabled=True)

    def on_ask_question(self):
        if self.state == "INSTRUCTION": self.speak_and_show("Yes, what's your question?", "Assistant")
        self.ui.set_button_state(self.state.lower(), enabled=False); question_text = self.ui.listen_for_question()
        self.ui.set_button_state(self.state.lower(), enabled=True)
        if question_text: self.ui.add_chat_message("Student", question_text); self.speak_and_show("Good question...", "Assistant"); answer = self.answer_question_with_llm(question_text); self.speak_and_show(answer, "Assistant")
        else: self.speak_and_show("I didn't catch that.", "Assistant")

    def get_llm_response(self, prompt, model="gemma2-9b-it"):
        try:
            completion = self.groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=model)
            return completion.choices[0].message.content
        except Exception as e: return f"LLM Error: {e}"

    def validate_with_llm(self, ocr_text, step):
        # Extract the solution keywords from the step dictionary
        solution_keywords = step.get('solution_keywords', [])
        prompt = self.validation_prompt_template.format(
            solution_keywords=solution_keywords,
            ocr_text=ocr_text
        )
        response_text = self.get_llm_response(prompt)
        print(response_text)
        # Try to extract JSON from the LLM response
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                result_json = json.loads(match.group(0))
                status = result_json.get("status", "").upper()
                return status == "COMPLETE"
            except Exception:
                pass
        # Fallback: check for "COMPLETE" in the response
        if "COMPLETE" in response_text.upper():
            return True
        return False

    def generate_hint_with_llm(self, ocr_text, step):
        prompt = self.hint_prompt_template.format(
            step_pedagogical_goal=step['pedagogical_goal'],
            ocr_text=ocr_text,
            student_condition=self.student_condition
        )
        response_text = self.get_llm_response(prompt)
        if "<speaking>" in response_text:
            return response_text.split("<speaking>")[1].split("</speaking>")[0].strip()
        return response_text

    def answer_question_with_llm(self, question):
        step_context = self.current_lesson[self.current_step_index]['question_text']
        prompt = f"AI Tutor...Student is '{self.student_condition}' on step '{step_context}'. They asked: '{question}'. Answer clearly."
        return self.get_llm_response(prompt)

    def speak_and_show(self, text, sender):
        self.ui.add_chat_message(sender, text)
        try:
            for old_file in (self.base_dir / "runs").glob("response_*.mp3"):
                try: os.remove(old_file)
                except OSError: pass

            timestamp = int(time.time() * 1000)
            audio_path = self.base_dir / "runs" / f"response_{timestamp}.mp3"
            tts = gTTS(text, lang='en')
            tts.save(audio_path)
            
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def speak(self, text):
        """TTS only, without adding to chat history."""
        try:
            for old_file in (self.base_dir / "runs").glob("response_*.mp3"):
                try: os.remove(old_file)
                except OSError: pass
            
            timestamp = int(time.time() * 1000)
            audio_path = self.base_dir / "runs" / f"response_{timestamp}.mp3"
            tts = gTTS(text, lang='en')
            tts.save(audio_path)
            
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"TTS Error: {e}")

    def stop_tts(self):
        """Stops any currently playing audio."""
        pygame.mixer.music.stop()

# ==============================================================================
#  4. APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
