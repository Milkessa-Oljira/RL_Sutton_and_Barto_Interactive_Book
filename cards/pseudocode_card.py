from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class PseudocodeCard(QWidget):
    def __init__(self, item):
        super().__init__()
        content = item.get("content", "No pseudocode provided.")
        text_edit = QTextEdit()
        text_edit.setPlainText(content)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d44;
                color: #e0e0ff;
                font-size: 14px;
                padding: 10px;
                border: none;
            }
        """)
        layout = QVBoxLayout()
        layout.addWidget(text_edit)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")