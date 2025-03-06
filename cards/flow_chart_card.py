from graphviz import Digraph
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os

class FlowChartCard(QWidget):
    def __init__(self, item):
        super().__init__()
        # Extract flowchart data
        flowchart_data = item.get("flowchart_data", {})
        nodes = flowchart_data.get("nodes", [])
        edges = flowchart_data.get("edges", [])

        # Create dynamic flowchart with graphviz
        dot = Digraph()
        for node in nodes:
            dot.node(node)
        for edge in edges:
            dot.edge(edge[0], edge[1])

        # Render to a temporary PNG file
        dot.format = 'png'
        dot.render('temp_flowchart', cleanup=True)

        # Load and display the flowchart
        pixmap = QPixmap('temp_flowchart.png').scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label = QLabel()
        label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")

        # Clean up temporary file
        os.remove('temp_flowchart.png')