from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import Qt

class InteractiveCodeCard(QWidget):
    def __init__(self, item):
        super().__init__()
        # Code editor
        self.code_editor = QTextEdit()
        self.code_editor.setPlainText(item.get("content", "# Write your code here"))
        self.code_editor.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d44;
                color: #e0e0ff;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                padding: 10px;
            }
        """)

        # Run button
        self.run_button = QPushButton("Run Code")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4a4a7d;
                color: #ffffff;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a5a9d;
            }
        """)

        # Output area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d44;
                color: #e0e0ff;
                font-size: 14px;
                padding: 10px;
            }
        """)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.code_editor)
        layout.addWidget(self.run_button)
        layout.addWidget(self.output_area)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")
        self.run_button.clicked.connect(self.run_code)

    def run_code(self):
        code = self.code_editor.toPlainText()
        output = ""
        try:
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            redirected_output = StringIO()
            sys.stdout = redirected_output
            exec(code)  # Full Python with RL library support
            sys.stdout = old_stdout
            output = redirected_output.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        self.output_area.setPlainText(output)