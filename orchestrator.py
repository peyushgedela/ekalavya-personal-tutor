import os
import time
import json
import pathlib
import threading

from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
import pygame

from PyQt6.QtCore import QTimer

from analysis import AnalysisWorker


class TutorOrchestrator:
    def __init__(self, main_window):
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
        # OverheadCamera is expected to be available at project root as OverheadCamera.py
        from OverheadCamera import OverheadCamera

        self.overhead_camera = OverheadCamera(ip_url=ip_camera_url, groq_client=self.groq_client, ocr_prompt=self.ocr_prompt_template)

        self.state = "IDLE"
        self.current_lesson = None
        self.current_step_index = 0
        self.attempt_counter = 0
        self.student_condition = "attentive"

        # Signal connections will be done when practice_widget is created

    def connect_practice_widget_signals(self):
        """Connect signals for the practice widget after it's created"""
        if self.ui.practice_widget:
            self.ui.practice_widget.check_work_requested.connect(self.on_check_work)
            self.ui.practice_widget.ask_question_requested.connect(self.on_ask_question)
            self.ui.practice_widget.student_condition_updated.connect(self.on_student_condition_update)

    def on_student_condition_update(self, condition: str):
        """Receives the new, richer condition string and updates the state."""
        if self.student_condition != condition:
            self.student_condition = condition
            print(f"Student condition updated to: {self.student_condition}")

    def start(self, filename: str | None = None):
        if not filename:
            filename = "lesson_remedial.json"
        self.load_lesson(filename)

    def load_lesson(self, filename):
        try:
            with open(self.base_dir / "lessons" / filename, 'r', encoding="utf-8") as f:
                self.current_lesson = json.load(f)
            self.current_step_index = 0
            self.process_current_step()
        except FileNotFoundError:
            self.ui.add_chat_message("Assistant", f"Error: Lesson file '{filename}' not found.")

    def process_current_step(self):
        if self.current_step_index >= len(self.current_lesson):
            self.speak_and_show("Great job! You've completed all the steps.", "Assistant")
            self.ui.set_button_state("practice", enabled=False)
            return
        step = self.current_lesson[self.current_step_index]
        self.ui.update_question(step['question_text'])

        svg_path = str(self.base_dir / "assets" / step['svg_file'])
        if os.path.exists(svg_path):
            self.ui.update_image(svg_path)
        else:
            # Try looking for other image formats
            for ext in ['.png', '.jpg', '.jpeg']:
                image_path = str(self.base_dir / "assets" / step['svg_file'].replace('.svg', ext))
                if os.path.exists(image_path):
                    self.ui.update_image(image_path)
                    break
            else:  # No image found in any format
                print(f"Warning: Image file not found for {step['svg_file']}")
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
            self.state = "PRACTICE"
            self.ui.set_button_state("practice")
            if self.attempt_counter == 0:
                self.speak_and_show("Okay, your turn...", "Assistant")

    def advance_step(self):
        if self.state == "INSTRUCTION":
            self.current_step_index += 1
            self.process_current_step()
        elif self.state == "PRACTICE":
            self.attempt_counter = 0
            self.current_step_index += 1
            self.process_current_step()

    def on_check_work(self):
        self.ui.set_button_state("practice", enabled=False)
        self.speak_and_show("Okay, let me check...", "Assistant")
        self.worker = AnalysisWorker(self.overhead_camera)
        self.thread = threading.Thread(target=self.worker.run)
        self.worker.analysis_complete.connect(self.on_analysis_complete)
        self.thread.start()

    def on_analysis_complete(self, result):
        if "error" in result:
            self.speak_and_show(f"Error: {result['error']}", "Assistant")
            self.ui.set_button_state("practice", enabled=True)
            return
        analysis = result.get("analysis", {})
        ocr_text = "\n".join(analysis.get("lines", []))
        if not ocr_text.strip():
            self.speak_and_show("I didn't see any writing.", "Assistant")
            self.ui.set_button_state("practice", enabled=True)
            return
        self.ui.add_chat_message("Assistant", f"I see: \"{ocr_text}\"")
        step = self.current_lesson[self.current_step_index]
        is_complete = self.validate_with_llm(ocr_text, step)
        if is_complete:
            self.speak_and_show("Perfect!", "Assistant")
            self.advance_step()
        else:
            self.attempt_counter += 1
            hint = self.generate_hint_with_llm(ocr_text, step)
            self.speak_and_show(hint, "Assistant")
            self.ui.set_button_state("practice", enabled=True)

    def on_ask_question(self):
        if self.state == "INSTRUCTION":
            self.speak_and_show("Yes, what's your question?", "Assistant")
        self.ui.set_button_state(self.state.lower(), enabled=False)
        question_text = self.ui.listen_for_question()
        self.ui.set_button_state(self.state.lower(), enabled=True)
        if question_text:
            self.ui.add_chat_message("Student", question_text)
            self.speak_and_show("Good question...", "Assistant")
            answer = self.answer_question_with_llm(question_text)
            self.speak_and_show(answer, "Assistant")
        else:
            self.speak_and_show("I didn't catch that.", "Assistant")

    def get_llm_response(self, prompt, model="meta-llama/llama-guard-4-12b"):
        try:
            completion = self.groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=model)
            return completion.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {e}"

    def validate_with_llm(self, ocr_text, step):
        solution_keywords = step.get('solution_keywords', [])
        prompt = self.validation_prompt_template.format(
            solution_keywords=solution_keywords,
            ocr_text=ocr_text
        )
        response_text = self.get_llm_response(prompt)
        print(response_text)
        match = __import__('re').search(r"\{.*\}", response_text, __import__('re').DOTALL)
        if match:
            try:
                result_json = json.loads(match.group(0))
                status = result_json.get("status", "").upper()
                return status == "COMPLETE"
            except Exception:
                pass
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
                try:
                    os.remove(old_file)
                except OSError:
                    pass

            timestamp = int(time.time() * 1000)
            audio_path = self.base_dir / "runs" / f"response_{timestamp}.mp3"
            tts = gTTS(text, lang='en')
            tts.save(audio_path)

            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"TTS Error: {e}")

    def speak(self, text):
        try:
            for old_file in (self.base_dir / "runs").glob("response_*.mp3"):
                try:
                    os.remove(old_file)
                except OSError:
                    pass

            timestamp = int(time.time() * 1000)
            audio_path = self.base_dir / "runs" / f"response_{timestamp}.mp3"
            tts = gTTS(text, lang='en')
            tts.save(audio_path)

            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"TTS Error: {e}")

    def stop_tts(self):
        pygame.mixer.music.stop()
