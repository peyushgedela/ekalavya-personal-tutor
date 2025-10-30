import cv2
import os
import time
import pathlib
from groq import Groq

class OverheadCamera:
    """
    Handles on-demand image capture from an IP camera and sends it for analysis.
    """
    def __init__(self, ip_url: str, groq_client: Groq, ocr_prompt: str, save_dir: str = "captures"):
        """
        Initializes the camera with the IP stream URL and the OCR prompt template.
        """
        self.ip_url = ip_url
        self.groq_client = groq_client
        self.ocr_prompt = ocr_prompt # Stores the prompt template
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def _capture_frame(self):
        """
        Connects to the IP camera, captures a single frame, and disconnects.
        """
        cap = cv2.VideoCapture(self.ip_url)
        if not cap.isOpened():
            print(f"Error: Could not open camera stream at {self.ip_url}")
            return False, None
        ret, frame = cap.read()
        cap.release()
        return ret, frame

    def _analyze_image_with_groq(self, image_path: str, encode_image, extract_json_from_llm_output):
        """
        Encodes the captured image and sends it to the Groq vision model for analysis.
        """
        base64_image = encode_image(image_path)
        try:
            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": self.ocr_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}],
                temperature=0.1, max_tokens=1024)
            result_text = completion.choices[0].message.content
            return {"analysis": extract_json_from_llm_output(result_text)}
        except Exception as e:
            print(f"Error during Groq API call: {e}")
            return {"error": str(e)}

    def capture_and_analyze(self, encode_image, extract_json_from_llm_output):
        """
        The main public method that orchestrates capturing, saving, analyzing, and cleaning up.
        """
        ret, frame = self._capture_frame()
        if not ret:
            return {"error": "Failed to capture image from IP camera."}
        
        # Save the frame to a temporary file
        image_path = str(self.save_dir / f"capture_{int(time.time())}.jpg")
        cv2.imwrite(image_path, frame)
        
        # Analyze the saved image
        result = self._analyze_image_with_groq(image_path, encode_image, extract_json_from_llm_output)
        
        # Clean up the temporary file
        try:
            os.remove(image_path)
        except OSError as e:
            print(f"Error deleting temporary image file: {e}")
            
        return result