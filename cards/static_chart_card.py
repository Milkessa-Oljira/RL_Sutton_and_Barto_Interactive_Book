import matplotlib.pyplot as plt
from io import BytesIO
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class StaticChartCard(QWidget):
    def __init__(self, item):
        super().__init__()
        
        # Create a styled container frame
        frame = QFrame(self)
        frame.setFrameShape(QFrame.StyledPanel)
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
        
        chart_type = item.get("chart_type", "")
        data = item.get("data", {})
        title = item.get("title", "")
        
        # Try to use a preferred style and fallback if not available.
        try:
            plt.style.use('seaborn-darkgrid')
        except Exception:
            plt.style.use('dark_background')
        
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        
        if chart_type == "line" and "x" in data and "y" in data:
            ax.plot(data["x"], data["y"], marker='o', linestyle='-', color='#4a90e2', linewidth=2)
        elif chart_type == "bar" and "categories" in data and "values" in data:
            ax.bar(data["categories"], data["values"], color='#e94e77')
        elif chart_type == "scatter" and "x" in data and "y" in data:
            ax.scatter(data["x"], data["y"], color='#8e44ad', s=50)
        else:
            ax.text(0.5, 0.5, "Invalid chart data", ha='center', va='center', fontsize=12, color='red')
        
        ax.set_title(title, fontsize=14, color='#ffffff', pad=15)
        ax.set_facecolor('#2a2a3d')
        fig.patch.set_facecolor('#2a2a3d')
        ax.tick_params(colors='#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#ffffff')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
