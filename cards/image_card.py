from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageCard(QWidget):
    def __init__(self, item):
        super().__init__()
        content = item.get("content", "images/default_image.png")
        pixmap = QPixmap(content).scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label = QLabel()
        label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")