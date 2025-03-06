from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem

class TableCard(QWidget):
    def __init__(self, item):
        super().__init__()
        headers = item.get("headers", [])
        rows = item.get("rows", [])
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(cell)))
        table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d44;
                color: #e0e0ff;
                border: none;
            }
            QHeaderView::section {
                background-color: #4a4a7d;
                color: #ffffff;
                padding: 5px;
            }
        """)
        layout = QVBoxLayout()
        layout.addWidget(table)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a3d; border-radius: 5px; margin: 5px;")