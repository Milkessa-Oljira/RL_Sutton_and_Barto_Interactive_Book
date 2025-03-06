from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class TextCard(QWidget):
    def __init__(self, item):
        super().__init__()
        # Extract data from JSON
        title = item.get("title", "")
        content = item.get("content", "No content provided.")
        title_style = item.get("title_style", {})
        content_style = item.get("content_style", {})

        # Set up layout
        layout = QVBoxLayout()

        # Add title if provided
        if title:
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignLeft)
            title_label.setStyleSheet(f"""
                font-family: {title_style.get('font', 'Arial')};
                font-size: {title_style.get('size', 16)}px;
                color: {title_style.get('color', '#ffffff')};
                padding: 5px;
            """)
            layout.addWidget(title_label)

        # Add content
        content_label = QLabel(content)
        content_label.setWordWrap(True)
        content_label.setAlignment(Qt.AlignTop)
        content_label.setStyleSheet(f"""
            font-family: {content_style.get('font', 'Arial')};
            font-size: {content_style.get('size', 14)}px;
            color: {content_style.get('color', '#e0e0ff')};
            padding: 10px;
        """)
        layout.addWidget(content_label)

        # Apply layout and card styling
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")