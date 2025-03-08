from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt

class AnimationCard(QWidget):
    def __init__(self, item):
        super().__init__()
        
        # Container frame with drop shadow
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
        
        # Load and display the animation
        content = item.get("content", "assets/default_animation.gif")
        self.movie = QMovie(content)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMovie(self.movie)
        self.movie.start()
        layout.addWidget(label)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
