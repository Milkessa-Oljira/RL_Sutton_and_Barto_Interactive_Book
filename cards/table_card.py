from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QFrame, QGraphicsDropShadowEffect, QHeaderView
from PyQt5.QtCore import Qt

class TableCard(QWidget):
    def __init__(self, item):
        super().__init__()
        # Create a container frame with drop shadow
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

        headers = item.get("headers", [])
        rows = item.get("rows", [])

        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(headers)

        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(cell)))

        # Automatically adjust column sizes based on content
        header_view = table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeToContents)

        layout.addWidget(table)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)
