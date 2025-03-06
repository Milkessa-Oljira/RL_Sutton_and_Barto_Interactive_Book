import matplotlib.pyplot as plt
from io import BytesIO
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap

class StaticChartCard(QWidget):
    def __init__(self, item):
        super().__init__()
        chart_type = item.get("chart_type", "")
        data = item.get("data", {})
        title = item.get("title", "")
        fig, ax = plt.subplots(figsize=(5, 3))
        if chart_type == "line" and "x" in data and "y" in data:
            ax.plot(data["x"], data["y"])
        elif chart_type == "bar" and "categories" in data and "values" in data:
            ax.bar(data["categories"], data["values"])
        elif chart_type == "scatter" and "x" in data and "y" in data:
            ax.scatter(data["x"], data["y"])
        else:
            ax.text(0.5, 0.5, "Invalid chart data", ha='center')
        ax.set_title(title)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        label = QLabel()
        label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")