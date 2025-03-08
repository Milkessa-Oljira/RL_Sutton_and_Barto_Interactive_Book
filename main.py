import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QVBoxLayout, QLabel, QComboBox, QHBoxLayout
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Universe")
        self.setMinimumSize(800, 600)
        self.current_concept = 1

        # Create a central widget with vertical layout
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        
        # Top bar for theme selection (positioned at the top-right)
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.theme_combo = QComboBox()
        # Available theme background options (light modes first, then dark modes)
        self.themes = ["#fffaf0", "#f0f8ff", "#f5f5dc", "#272822", "#2a2a3d"]
        self.theme_combo.addItems(self.themes)
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        top_bar.addWidget(self.theme_combo)
        central_layout.addLayout(top_bar)
        
        # Scroll area for the concept content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.scroll_area.setWidget(self.content_widget)
        central_layout.addWidget(self.scroll_area)
        
        self.setCentralWidget(central_widget)
        self.loadConcept(self.current_concept)
        self.change_theme()  # apply initial theme

    def loadConcept(self, concept_number):
        # Clear previous content
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Load the JSON file for the given concept number
        try:
            with open(f"concepts/concept_{concept_number}.json", "r") as f:
                concept_data = json.load(f)
        except FileNotFoundError:
            self.content_layout.addWidget(QLabel(f"Concept file concept_{concept_number}.json not found."))
            return
        except json.JSONDecodeError:
            self.content_layout.addWidget(QLabel(f"Invalid JSON format in concept_{concept_number}.json."))
            return

        # Create and add each card widget based on its type
        for item in concept_data:
            card_type = item.get("type")
            if card_type in card_types:
                card_class = card_types[card_type]
                card_widget = card_class(item)
                self.content_layout.addWidget(card_widget)
            else:
                self.content_layout.addWidget(QLabel(f"Unknown card type: {card_type}"))

    def change_theme(self):
        # Get the selected theme background color
        selected = self.theme_combo.currentText()
        # Determine contrasting text color based on light vs dark backgrounds
        light_themes = ["#fffaf0", "#f0f8ff", "#f5f5dc"]
        if selected in light_themes:
            text_color = "#333333"
        else:
            text_color = "#e0e0e0"
        # Update global stylesheet for the main window and common widgets
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {selected};
                color: {text_color};
                font-family: 'Georgia', serif;
            }}
            QLabel, QTextBrowser, QTextEdit {{
                color: {text_color};
            }}
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
