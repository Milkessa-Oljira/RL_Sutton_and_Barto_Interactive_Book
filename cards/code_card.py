from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextBrowser
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

class CodeCard(QWidget):
    def __init__(self, item):
        super().__init__()
        language = item.get("language", "text")
        content = item.get("content", "No code provided.")
        try:
            lexer = get_lexer_by_name(language)
        except:
            lexer = get_lexer_by_name("text")
        formatter = HtmlFormatter(style='monokai')
        highlighted_code = highlight(content, lexer, formatter)
        text_browser = QTextBrowser()
        text_browser.setHtml(highlighted_code)
        text_browser.setStyleSheet("background-color: #272822; border: none; padding: 10px;")
        layout = QVBoxLayout()
        layout.addWidget(text_browser)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")