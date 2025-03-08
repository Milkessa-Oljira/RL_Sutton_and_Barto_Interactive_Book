from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QFrame, QGraphicsDropShadowEffect, QSizePolicy
from PyQt5.QtCore import Qt

class PseudocodeCard(QWidget):
    def __init__(self, item):
        super().__init__()
        
        # Create a container frame for improved aesthetics
        frame = QFrame(self)
        frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a3d;
                border: 2px solid #444;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setOffset(3, 3)
        shadow.setColor(Qt.black)
        frame.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Get pseudocode content from the JSON data
        content = item.get("content", "No pseudocode provided.")
        pseudocode_edit = QTextEdit()
        pseudocode_edit.setPlainText(content)
        pseudocode_edit.setReadOnly(True)
        pseudocode_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d44;
                color: #e0e0ff;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        layout.addWidget(pseudocode_edit)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
