from PyQt6.QtCore import QObject, pyqtSignal

from utils import encode_image, extract_json_from_llm_output


class AnalysisWorker(QObject):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, overhead_camera_instance):
        super().__init__()
        self.camera = overhead_camera_instance

    def run(self):
        # overhead_camera_instance is expected to implement capture_and_analyze
        result = self.camera.capture_and_analyze(encode_image, extract_json_from_llm_output)
        self.analysis_complete.emit(result)
