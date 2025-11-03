from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, 
                             QPushButton, QScrollArea, QFrame, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import os
import json

class TopicsWidget(QWidget):
    """Displays available topics for the student to choose from."""
    topic_selected = pyqtSignal(str)  # Emits the selected topic folder path

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header Section
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #00519E; border-radius: 15px; margin: 10px;")
        header_layout = QVBoxLayout(header_frame)
        
        subject_label = QLabel("Class 9 - Mathematics")
        subject_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        subject_label.setStyleSheet("color: white;")
        subject_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        chapter_label = QLabel("Lines and Angles")
        chapter_label.setFont(QFont("Segoe UI", 18))
        chapter_label.setStyleSheet("color: #E5E5EA;")
        chapter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(subject_label)
        header_layout.addWidget(chapter_label)
        layout.addWidget(header_frame)

        # Topics Section
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea { 
                border: none;
                background-color: transparent;
            }
        """)
        
        topics_widget = QWidget()
        topics_layout = QGridLayout(topics_widget)
        topics_layout.setSpacing(20)
        
        # Get topics from lessons directory
        lessons_path = os.path.join(os.path.dirname(__file__), 'lessons')
        topics = [d for d in os.listdir(lessons_path) 
                 if os.path.isdir(os.path.join(lessons_path, d)) and d != '__pycache__']
        
        # Create topic buttons
        for idx, topic in enumerate(topics):
            if topic == '__pycache__':
                continue
                
            topic_button = QPushButton(topic.replace('_', ' '))
            topic_button.setFont(QFont("Segoe UI", 14))
            topic_button.setMinimumHeight(100)
            topic_button.setStyleSheet("""
                QPushButton {
                    background-color: #FFFFFF;
                    color: #2C3E50;
                    border: 2px solid #E5E5EA;
                    border-radius: 15px;
                    padding: 15px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #F8F9FA;
                    border: 2px solid #00519E;
                }
            """)
            
            # Store the topic path as a property of the button
            topic_button.setProperty("topic_path", os.path.join(lessons_path, topic))
            topic_button.clicked.connect(lambda checked, btn=topic_button: 
                                      self.topic_selected.emit(btn.property("topic_path")))
            
            # Arrange buttons in a 2-column grid
            row = idx // 2
            col = idx % 2
            topics_layout.addWidget(topic_button, row, col)
        
        scroll_area.setWidget(topics_widget)
        layout.addWidget(scroll_area)