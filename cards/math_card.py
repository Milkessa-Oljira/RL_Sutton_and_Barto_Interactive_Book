import matplotlib.pyplot as plt
from io import BytesIO
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap

class MathCard(QWidget):
    def __init__(self, item):
        super().__init__()
        # Extract data
        content = item.get("content", "")
        text_color = item.get("text_color", "#ffffff")
        background_color = item.get("background_color", "#2a2a3d")

        # Render LaTeX with matplotlib
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.5, 0.5, f"${content}$", fontsize=16, ha='center', va='center', color=text_color)
        ax.set_facecolor(background_color)
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)

        # Display the rendered image
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        label = QLabel()
        label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        self.setStyleSheet(f"background-color: {background_color}; border-radius: 5px; margin: 5px;")