import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QDialog, QScrollArea, QWidget, QVBoxLayout, QLabel, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from cards.text_card import TextCard
from cards.math_card import MathCard
from cards.static_chart_card import StaticChartCard
from cards.image_card import ImageCard
from cards.code_card import CodeCard
from cards.pseudocode_card import PseudocodeCard
from cards.interactive_code_card import InteractiveCodeCard
from cards.animation_card import AnimationCard
from cards.table_card import TableCard
from cards.flow_chart_card import FlowChartCard

# Mapping of card types to their respective classes
card_types = {
    "text": TextCard,
    "math": MathCard,
    "static_chart": StaticChartCard,
    "image": ImageCard,
    "code": CodeCard,
    "pseudocode": PseudocodeCard,
    "interactive_code": InteractiveCodeCard,
    "animation": AnimationCard,
    "table": TableCard,
    "flow_chart": FlowChartCard
}

class ConceptItem(QGraphicsPixmapItem):
    """Custom QGraphicsPixmapItem for clickable concept icons."""
    def __init__(self, concept_name, parent=None):
        super().__init__(parent)
        self.concept_name = concept_name
        self.setCursor(Qt.PointingHandCursor)  # Show hand cursor on hover

    def mousePressEvent(self, event):
        """Open the concept window when clicked."""
        concept_window = ConceptWindow(self.concept_name)
        concept_window.exec_()

class ConceptWindow(QDialog):
    """Dialog window to display a concept's content using cards."""
    def __init__(self, concept_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{concept_name.replace('_', ' ').title()} - RL Universe")
        self.setMinimumSize(600, 400)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2f;
                color: #ffffff;
                font-family: 'Arial', sans-serif;
            }
            QScrollArea {
                border: none;
            }
        """)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout()

        try:
            with open(f"concepts/{concept_name}.json", "r") as f:
                concept_data = json.load(f)
        except FileNotFoundError:
            layout.addWidget(QLabel("Concept file not found."))
        except json.JSONDecodeError:
            layout.addWidget(QLabel("Invalid JSON format in concept file."))
        else:
            for item in concept_data:
                card_type = item.get("type")
                if card_type in card_types:
                    card_class = card_types[card_type]
                    card_widget = card_class(item)
                    layout.addWidget(card_widget)
                else:
                    layout.addWidget(QLabel(f"Unknown card type: {card_type}"))

        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

class MainWindow(QMainWindow):
    """Main window displaying the universe visualization."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Universe")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d0d1a;
            }
        """)

        view = QGraphicsView()
        scene = QGraphicsScene()
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        try:
            with open("universe.json", "r") as f:
                universe_data = json.load(f)
        except FileNotFoundError:
            scene.addText("universe.json not found.")
        except json.JSONDecodeError:
            scene.addText("Invalid JSON in universe.json.")
        else:
            background = QPixmap(universe_data.get("background", "images/default_background.png"))
            scene.setSceneRect(0, 0, background.width(), background.height())
            scene.addPixmap(background)
            for concept in universe_data.get("concepts", []):
                icon_path = concept.get("icon", "images/default_star.png")
                icon = QPixmap(icon_path)
                item = ConceptItem(concept["name"])
                item.setPixmap(icon.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                item.setPos(concept["x"], concept["y"])
                item.setToolTip(concept["name"].replace("_", " ").title())
                scene.addItem(item)

        view.setScene(scene)
        self.setCentralWidget(view)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())