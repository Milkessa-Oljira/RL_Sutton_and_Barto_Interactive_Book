import fitz  # PyMuPDF
import pygame
from pygame.locals import *
import networkx as nx
import math
import sys
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import io

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
MAP_CENTER_X = WINDOW_WIDTH // 3  # Map on the left third
MAP_CENTER_Y = WINDOW_HEIGHT // 2
TEXT_AREA_X = WINDOW_WIDTH // 3 + 50
TEXT_AREA_Y = 50
TEXT_AREA_WIDTH = 2 * WINDOW_WIDTH // 3 - 100
TEXT_AREA_HEIGHT = WINDOW_HEIGHT - 100
FONT_SIZE = 20
NODE_RADIUS = 10
EDGE_COLOR = (150, 150, 150)
NODE_COLOR = (0, 120, 255)
SELECTED_NODE_COLOR = (255, 50, 50)
TEXT_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)

# Node class for TOC entries
class Node:
    def __init__(self, id: int, title: str, level: int, page_start: int):
        self.id = id
        self.title = title.strip()
        self.level = level
        self.page_start = page_start - 1  # PyMuPDF uses 0-based indexing
        self.page_end: Optional[int] = None
        self.position: Optional[Tuple[float, float]] = None

# PDF Processing
def extract_toc_and_doc(pdf_path: str) -> Tuple[List, fitz.Document]:
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    if not toc:
        raise ValueError("PDF has no table of contents.")
    return toc, doc

def build_graph(toc: List) -> Tuple[nx.Graph, Dict[int, Node]]:
    G = nx.Graph()
    nodes = {}
    node_id = 0
    stack = []
    
    for entry in toc:
        level, title, page = entry
        node = Node(node_id, title, level, page)
        G.add_node(node_id)
        nodes[node_id] = node
        node_id += 1
        
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            parent_id = stack[-1].id
            G.add_edge(parent_id, node_id)
        stack.append(node)
    
    return G, nodes

def set_page_ranges(G: nx.Graph, nodes: Dict[int, Node], doc: fitz.Document):
    def traverse(node_id: int, next_start: int):
        node = nodes[node_id]
        children = list(G.neighbors(node_id))
        if not children:
            node.page_end = min(next_start, doc.page_count)
            return
        for i, child_id in enumerate(children):
            next_node_start = nodes[children[i + 1]].page_start if i + 1 < len(children) else next_start
            traverse(child_id, next_node_start)
            if i == 0:
                node.page_start = nodes[child_id].page_start
        node.page_end = min(next_start, doc.page_count)
    
    root_id = next(iter(G.nodes))  # Assuming the first node is the root
    traverse(root_id, doc.page_count + 1)

def extract_text(doc: fitz.Document, start_page: int, end_page: int) -> str:
    text = ""
    for page_num in range(start_page, end_page):
        text += doc[page_num].get_text("text") + "\n"
    return text.strip()

# Layout
def compute_layout(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    pos = nx.spring_layout(G, k=1.0, iterations=50)
    return pos

# Mathematical Rendering
def render_equation(equation: str) -> pygame.Surface:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f"${equation}$", fontsize=12, ha='center', va='center')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = pygame.image.load(buf)
    plt.close(fig)
    return image

def extract_and_render_equations(text: str) -> List[Tuple[str, Optional[pygame.Surface]]]:
    parts = []
    while True:
        start = text.find('$')
        if start == -1:
            parts.append((text, None))
            break
        end = text.find('$', start + 1)
        if end == -1:
            parts.append((text, None))
            break
        equation = text[start + 1:end]
        image = render_equation(equation)
        parts.append((text[:start], None))
        parts.append((equation, image))
        text = text[end + 1:]
    return parts

# User Interface
class KnowledgeMapUI:
    def __init__(self, pdf_path: str):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Interactive Knowledge Map")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, FONT_SIZE)
        
        self.toc, self.doc = extract_toc_and_doc(pdf_path)
        self.G, self.nodes = build_graph(self.toc)
        set_page_ranges(self.G, self.nodes, self.doc)
        self.positions = compute_layout(self.G)
        for node_id, pos in self.positions.items():
            self.nodes[node_id].position = pos
        
        self.cx = 0.0
        self.cy = 0.0
        self.scale = 1.0
        self.dragging = False
        self.last_mouse_pos = None
        self.selected_node: Optional[Node] = None
        self.text_cache: Dict[int, str] = {}

    def transform_position(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        x, y = pos
        screen_x = (x - self.cx) * self.scale + MAP_CENTER_X
        screen_y = (y - self.cy) * self.scale + MAP_CENTER_Y
        return int(screen_x), int(screen_y)

    def inverse_transform(self, screen_pos: Tuple[int, int]) -> Tuple[float, float]:
        screen_x, screen_y = screen_pos
        x = (screen_x - MAP_CENTER_X) / self.scale + self.cx
        y = (screen_y - MAP_CENTER_Y) / self.scale + self.cy
        return x, y

    def draw_map(self):
        for edge in self.G.edges:
            node1_pos = self.transform_position(self.nodes[edge[0]].position)
            node2_pos = self.transform_position(self.nodes[edge[1]].position)
            pygame.draw.line(self.screen, EDGE_COLOR, node1_pos, node2_pos, 1)
        
        for node_id, node in self.nodes.items():
            pos = self.transform_position(node.position)
            color = SELECTED_NODE_COLOR if node == self.selected_node else NODE_COLOR
            pygame.draw.circle(self.screen, color, pos, NODE_RADIUS)
            if self.scale > 0.5:
                text_surface = self.font.render(node.title[:20], True, TEXT_COLOR)
                self.screen.blit(text_surface, (pos[0] + 10, pos[1] - 10))

    def draw_text(self, text: str):
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, (TEXT_AREA_X - 10, TEXT_AREA_Y - 10, 
                                                         TEXT_AREA_WIDTH + 20, TEXT_AREA_HEIGHT + 20))
        parts = extract_and_render_equations(text)
        y = TEXT_AREA_Y
        for part, image in parts:
            if image:
                self.screen.blit(image, (TEXT_AREA_X, y))
                y += image.get_height() + 5
            else:
                lines = self.wrap_text(part, TEXT_AREA_WIDTH)
                for line in lines:
                    text_surface = self.font.render(line, True, TEXT_COLOR)
                    self.screen.blit(text_surface, (TEXT_AREA_X, y))
                    y += FONT_SIZE
                    if y > TEXT_AREA_Y + TEXT_AREA_HEIGHT:
                        break

    def wrap_text(self, text: str, max_width: int) -> List[str]:
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            width, _ = self.font.size(test_line)
            if width <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def find_closest_node(self, pos: Tuple[float, float]) -> Optional[Node]:
        min_dist = float('inf')
        closest = None
        for node in self.nodes.values():
            dx = node.position[0] - pos[0]
            dy = node.position[1] - pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
                closest = node
        return closest if min_dist * self.scale < NODE_RADIUS * 2 else None

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = self.inverse_transform(event.pos)
                        node = self.find_closest_node(mouse_pos)
                        if node:
                            self.selected_node = node
                            if node.id not in self.text_cache:
                                self.text_cache[node.id] = extract_text(self.doc, node.page_start, node.page_end or self.doc.page_count)
                    elif event.button == 3:
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 3:
                        self.dragging = False
                elif event.type == MOUSEMOTION:
                    if self.dragging:
                        dx = event.pos[0] - self.last_mouse_pos[0]
                        dy = event.pos[1] - self.last_mouse_pos[1]
                        self.cx -= dx / self.scale
                        self.cy -= dy / self.scale
                        self.last_mouse_pos = event.pos
                elif event.type == MOUSEWHEEL:
                    zoom_factor = 1.1 if event.y > 0 else 0.9
                    old_scale = self.scale
                    self.scale *= zoom_factor
                    mouse_x, mouse_y = self.inverse_transform(pygame.mouse.get_pos())
                    self.cx = mouse_x - (mouse_x - self.cx) * (self.scale / old_scale)
                    self.cy = mouse_y - (mouse_y - self.cy) * (self.scale / old_scale)
                    self.scale = max(0.1, min(self.scale, 10))

            self.screen.fill(BACKGROUND_COLOR)
            self.draw_map()
            if self.selected_node and self.selected_node.id in self.text_cache:
                self.draw_text(self.text_cache[self.selected_node.id])
            pygame.display.flip()
            self.clock.tick(60)

# Run the application
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python knowledge_map.py <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    ui = KnowledgeMapUI(pdf_path)
    ui.run()