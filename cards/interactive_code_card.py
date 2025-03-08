from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QFrame, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt

class InteractiveCodeCard(QWidget):
    def __init__(self, item):
        super().__init__()
        
        # Container frame with drop shadow for a modern look
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
        
        # Code editor setup
        self.code_editor = QTextEdit()
        self.code_editor.setPlainText(item.get("content", "# Write your code here"))
        self.code_editor.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d44;
                color: #e0e0ff;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.code_editor)
        
        # Run button setup with horizontal alignment to the right
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.run_button = QPushButton("Run Code")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4a4a7d;
                color: #ffffff;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5a5a9d;
            }
        """)
        self.run_button.clicked.connect(self.run_code)
        button_layout.addWidget(self.run_button)
        layout.addLayout(button_layout)
        
        # Output area setup
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d44;
                color: #e0e0ff;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.output_area)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
    
    def run_code(self):
        code = self.code_editor.toPlainText()
        output = ""
        try:
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            redirected_output = StringIO()
            sys.stdout = redirected_output
            exec(code)  # Caution: using exec; ensure proper sandboxing for production!
            sys.stdout = old_stdout
            output = redirected_output.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        self.output_area.setPlainText(output)
