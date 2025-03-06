from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QMovie

class AnimationCard(QWidget):
    def __init__(self, item):
        super().__init__()
        content = item.get("content", "images/default_animation.gif")
        self.movie = QMovie(content)
        label = QLabel()
        label.setMovie(self.movie)
        self.movie.start()
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")