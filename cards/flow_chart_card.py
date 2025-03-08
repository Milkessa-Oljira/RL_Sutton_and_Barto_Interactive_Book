from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFrame, QGraphicsDropShadowEffect, QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsTextItem
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF

import re

# Custom NodeItem class: a movable circle with centered text
class NodeItem(QGraphicsItem):
    def __init__(self, text, radius=50):
        super().__init__()
        self.radius = radius
        self.text = text
        self.text_item = QGraphicsTextItem(text, self)
        self.text_item.setDefaultTextColor(Qt.black)
        # Allow the text to wrap and be centered within the node
        self.text_item.setTextWidth(radius * 1.5)
        # Center the text approximately; adjust as needed
        self.text_item.setPos(-radius * 0.75, -radius * 0.5)
        self.edges = []
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)

    def boundingRect(self):
        return QRectF(-self.radius, -self.radius, self.radius * 2, self.radius * 2)

    def paint(self, painter, option, widget):
        painter.setBrush(Qt.white)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(self.boundingRect())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for edge in self.edges:
                edge.updatePosition()
        return super().itemChange(change, value)

# Custom EdgeItem that connects two NodeItems
class EdgeItem(QGraphicsItem):
    def __init__(self, source_node, dest_node):
        super().__init__()
        self.source = source_node
        self.dest = dest_node
        self.source.edges.append(self)
        self.dest.edges.append(self)
        self.setZValue(-1)

    def boundingRect(self):
        extra = 5
        line = self._line()
        return QRectF(line[0], line[1]).normalized().adjusted(-extra, -extra, extra, extra)

    def _line(self):
        # Returns a tuple of (start point, end point)
        return (self.source.pos(), self.dest.pos())

    def updatePosition(self):
        self.prepareGeometryChange()
        self.update()

    def paint(self, painter, option, widget):
        start, end = self._line()
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(start, end)

class FlowChartCard(QWidget):
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

        # Create QGraphicsScene and QGraphicsView for the interactive flowchart
        self.scene = QGraphicsScene()
        view = QGraphicsView(self.scene)
        view.setRenderHint(QPainter.Antialiasing)
        view.setStyleSheet("background: transparent; border: none;")

        # Retrieve flowchart data from JSON
        flowchart_data = item.get("flowchart_data", {})
        nodes = flowchart_data.get("nodes", [])
        edges = flowchart_data.get("edges", [])

        # Create NodeItems with initial positions
        self.node_items = {}
        x, y, step = 100, 100, 150
        for node in nodes:
            node_item = NodeItem(node, radius=50)
            node_item.setPos(x, y)
            self.scene.addItem(node_item)
            self.node_items[node] = node_item
            x += step
            if x > 400:
                x = 100
                y += step

        # Create EdgeItems connecting the nodes
        for edge in edges:
            if len(edge) >= 2:
                from_node, to_node = edge[0], edge[1]
                if from_node in self.node_items and to_node in self.node_items:
                    edge_item = EdgeItem(self.node_items[from_node], self.node_items[to_node])
                    self.scene.addItem(edge_item)

        layout.addWidget(view)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
