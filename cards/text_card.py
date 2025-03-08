from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt

class TextCard(QWidget):
    def __init__(self, item):
        super().__init__()
        
        # Read customization options from the JSON, with defaults
        background_color = item.get("background_color", "#2a2a3d")
        border_color = item.get("border_color", "#444444")
        border_radius = item.get("border_radius", "10px")
        padding = item.get("padding", "15px")
        
        # Create a styled container frame
        frame = QFrame(self)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {background_color};
                border: 2px solid {border_color};
                border-radius: {border_radius};
                margin: 5px;
            }}
        """)
        
        # Add a drop shadow for a modern, paper-like feel
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setOffset(3, 3)
        shadow.setColor(Qt.black)
        frame.setGraphicsEffect(shadow)
        
        # Internal layout for the card with customizable padding and spacing
        layout = QVBoxLayout(frame)
        # Remove "px" and convert to integer for margins
        pad = int(padding.rstrip('px'))
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(10)

        # Read text card content and styling options
        title = item.get("title", "")
        subtitle = item.get("subtitle", "")
        content = item.get("content", "No content provided.")
        
        title_font = item.get("title_font", "Arial")
        title_size = item.get("title_size", 20)
        title_color = item.get("title_color", "#ffffff")
        
        content_font = item.get("content_font", "Arial")
        content_size = item.get("content_size", 14)
        content_color = item.get("content_color", "#e0e0ff")
        
        # Optionally, add a subtitle if provided
        if title:
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignLeft)
            title_label.setStyleSheet(
                f"font-family: {title_font}; font-size: {title_size}px; color: {title_color}; font-weight: bold;"
            )
            layout.addWidget(title_label)
        
        if subtitle:
            subtitle_font = item.get("subtitle_font", "Arial")
            subtitle_size = item.get("subtitle_size", 16)
            subtitle_color = item.get("subtitle_color", "#cccccc")
            subtitle_label = QLabel(subtitle)
            subtitle_label.setAlignment(Qt.AlignLeft)
            subtitle_label.setStyleSheet(
                f"font-family: {subtitle_font}; font-size: {subtitle_size}px; color: {subtitle_color}; font-style: italic;"
            )
            layout.addWidget(subtitle_label)
        
        content_label = QLabel(content)
        content_label.setWordWrap(True)
        content_label.setAlignment(Qt.AlignTop)
        content_label.setStyleSheet(
            f"font-family: {content_font}; font-size: {content_size}px; color: {content_color};"
        )
        layout.addWidget(content_label)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
