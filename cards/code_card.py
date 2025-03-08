from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout, QFrame, QGraphicsDropShadowEffect, QToolTip
from PyQt5.QtCore import Qt, QPoint
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

class CodeCard(QWidget):
    def __init__(self, item):
        super().__init__()
        
        # Create a container frame for better styling
        frame = QFrame(self)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                background-color: #272822;
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
        
        main_layout = QVBoxLayout(frame)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        language = item.get("language", "text")
        content = item.get("content", "No code provided.")
        try:
            lexer = get_lexer_by_name(language)
        except Exception:
            lexer = get_lexer_by_name("text")
        
        # Use Pygments with line numbers and inline styles for better integration
        formatter = HtmlFormatter(style='monokai', linenos=True, noclasses=True)
        highlighted_code = highlight(content, lexer, formatter)
        
        text_browser = QTextBrowser()
        text_browser.setHtml(highlighted_code)
        text_browser.setStyleSheet("background-color: transparent; border: none;")
        text_browser.setReadOnly(True)
        
        # Add a top-right "Copy" button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        copy_button = QPushButton("Copy")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #4a4a7d;
                color: #ffffff;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a5a9d;
            }
        """)
        copy_button.clicked.connect(lambda: self.copyCode(content))
        button_layout.addWidget(copy_button)
        
        main_layout.addLayout(button_layout)
        main_layout.addWidget(text_browser)
        
        layout = QVBoxLayout(self)
        layout.addWidget(frame)
        self.setLayout(layout)
        self.copy_button = copy_button
    
    def copyCode(self, code):
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(code)
        QToolTip.showText(self.copy_button.mapToGlobal(QPoint(0, 0)), "Code Copied!")
