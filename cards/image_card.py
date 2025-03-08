from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageCard(QWidget):
    def __init__(self, item):
        super().__init__()
        
        # Create a styled frame container
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Load the image from the JSON-specified file path
        content = item.get("content", "assets/default_image.png")
        pixmap = QPixmap(content)
        # Optionally scale the image for consistency
        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
