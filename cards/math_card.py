import matplotlib.pyplot as plt
from io import BytesIO
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MathCard(QWidget):
    def __init__(self, item):
        super().__init__()
        
        # Create a styled container frame
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
        
        # Retrieve LaTeX content and style parameters from JSON
        content = item.get("content", "")
        text_color = item.get("text_color", "#ffffff")
        background_color = item.get("background_color", "#2a2a3d")
        
        # Render LaTeX using matplotlib with a modern, minimal style
        fig, ax = plt.subplots(figsize=(4, 1), dpi=100)
        ax.text(0.5, 0.5, f"${content}$", fontsize=16, ha='center', va='center', color=text_color)
        ax.set_facecolor(background_color)
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        
        # Load the rendered image into a QPixmap and display it
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
