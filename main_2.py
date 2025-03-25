import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import numpy as np
import threading
import time
from PIL import Image, ImageTk
import fitz  # PyMuPDF for PDF handling
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from nltk.tokenize import sent_tokenize
import nltk
import re
import math
from scipy.signal import savgol_filter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Load NLP model
nlp = spacy.load("en_core_web_sm")

class ContentAnalyzer:
    def __init__(self):
        # Load pre-trained models for text analysis
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.pca = PCA(n_components=3)
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        document = fitz.open(pdf_path)
        text = ""
        for page in document:
            text += page.get_text()
        return text
    
    def extract_text_from_epub(self, epub_path):
        """Extract text from EPUB file"""
        book = epub.read_epub(epub_path)
        text = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text()
        return text
    
    def extract_text(self, file_path):
        """Extract text from either PDF or EPUB"""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.epub'):
            return self.extract_text_from_epub(file_path)
        else:
            raise ValueError("Unsupported file format")
    
    def split_into_sections(self, text):
        """Split text into logical sections"""
        # Basic section splitting by headers or chapter markers
        sections = re.split(r'\n\s*(?:CHAPTER|Chapter|SECTION|Section)\s+\w+', text)
        # Remove empty sections
        sections = [s.strip() for s in sections if s.strip()]
        return sections
    
    def analyze_complexity(self, text):
        """Analyze text complexity"""
        doc = nlp(text)
        
        # Metrics for complexity
        avg_word_length = sum(len(token.text) for token in doc if not token.is_punct) / max(1, sum(1 for token in doc if not token.is_punct))
        avg_sentence_length = sum(len(list(sent)) for sent in doc.sents) / max(1, len(list(doc.sents)))
        
        # Count of complex words (more than 3 syllables)
        complex_words = sum(1 for token in doc if len(token.text) > 8 and token.is_alpha)
        
        # Lexical diversity (unique words / total words)
        total_words = len([token for token in doc if token.is_alpha])
        unique_words = len(set([token.text.lower() for token in doc if token.is_alpha]))
        lexical_diversity = unique_words / max(1, total_words)
        
        # Named entity density
        named_entities = len(doc.ents)
        named_entity_density = named_entities / max(1, total_words)
        
        # Combine metrics into complexity score (0-1)
        complexity = (
            0.2 * min(1, avg_word_length / 10) +
            0.2 * min(1, avg_sentence_length / 40) +
            0.2 * min(1, complex_words / max(1, total_words * 0.1)) +
            0.2 * lexical_diversity +
            0.2 * min(1, named_entity_density * 10)
        )
        
        return complexity
    
    def analyze_content_density(self, text):
        """Analyze content density"""
        doc = nlp(text)
        
        # Count key information markers
        fact_markers = ["is", "are", "was", "were", "defines", "consists of"]
        concept_markers = ["theory", "principle", "concept", "framework", "model"]
        
        fact_count = sum(1 for token in doc if token.lemma_ in fact_markers)
        concept_count = sum(1 for token in doc if token.text.lower() in concept_markers)
        
        # Count numbers which often indicate data
        number_count = sum(1 for token in doc if token.like_num)
        
        # Count technical terms
        technical_terms = sum(1 for token in doc if token.is_alpha and token.is_title)
        
        total_words = len([token for token in doc if token.is_alpha])
        
        # Combine metrics into density score (0-1)
        density = min(1, (
            0.3 * fact_count / max(1, total_words * 0.05) +
            0.3 * concept_count / max(1, total_words * 0.01) +
            0.2 * number_count / max(1, total_words * 0.02) +
            0.2 * technical_terms / max(1, total_words * 0.05)
        ))
        
        return density
    
    def identify_themes(self, text):
        """Identify main themes in the text"""
        # Use TF-IDF to identify important words
        vectorized = self.tfidf.fit_transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        
        # Get top words
        scores = zip(feature_names, np.asarray(vectorized.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Group into themes
        top_words = [word for word, score in sorted_scores[:50]]
        
        # Simple theme clustering - in real app, use clustering algorithm
        themes = []
        doc = nlp(text)
        
        # Find sentences with top words
        important_sentences = []
        sentences = sent_tokenize(text)
        for sentence in sentences:
            if any(word in sentence.lower() for word in top_words[:20]):
                important_sentences.append(sentence)
        
        # Extract themes from important sentences
        for sentence in important_sentences[:10]:  # Limit to top 10 sentences
            s_doc = nlp(sentence)
            for chunk in s_doc.noun_chunks:
                if chunk.root.text.lower() in top_words:
                    themes.append(chunk.text)
        
        # Deduplicate themes
        themes = list(set(themes))[:5]  # Limit to top 5 themes
        
        return themes
    
    def generate_terrain_data(self, sections):
        """Generate terrain data for visualization"""
        # Initialize terrain arrays
        section_count = len(sections)
        complexity_profile = np.zeros(section_count)
        density_profile = np.zeros(section_count)
        theme_strength = np.zeros((section_count, 5))  # Assuming 5 main themes
        
        # Analyze each section
        themes = []
        for i, section in enumerate(sections):
            complexity_profile[i] = self.analyze_complexity(section)
            density_profile[i] = self.analyze_content_density(section)
            
            # If it's the first section, identify themes
            if i == 0:
                themes = self.identify_themes(" ".join(sections))
            
            # Analyze theme strength in this section
            section_themes = self.identify_themes(section)
            for j, theme in enumerate(themes[:5]):
                theme_strength[i, j] = 1 if theme in section_themes else 0
        
        # Smooth the profiles for better visualization
        complexity_profile = savgol_filter(complexity_profile, min(7, len(complexity_profile) - (len(complexity_profile) % 2 - 1)), 3)
        density_profile = savgol_filter(density_profile, min(7, len(density_profile) - (len(density_profile) % 2 - 1)), 3)
        
        return complexity_profile, density_profile, theme_strength, themes
    
    def generate_reading_landscape(self, text):
        """Generate the full reading landscape"""
        sections = self.split_into_sections(text)
        if not sections:
            sections = [text]  # If no clear sections, treat as one section
        
        # Generate terrain data
        complexity, density, theme_strength, themes = self.generate_terrain_data(sections)
        
        # Create the landscape model
        length = len(sections)
        width = 10  # Width of the terrain
        
        # Create mesh grid
        x = np.linspace(0, length, length * 10)
        y = np.linspace(-width/2, width/2, width * 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Generate height map based on complexity and density
        for i in range(length):
            # Base height is complexity
            base_height = complexity[i]
            
            # Add variations based on density
            variation = density[i]
            
            # Apply to Z within this section
            section_start = i * 10
            section_end = (i + 1) * 10
            
            for j in range(len(y)):
                # Calculate distance from center
                dist = abs(y[j]) / (width/2)
                
                # Higher in the center, lower at edges
                edge_factor = 1 - dist**2
                
                # Create mountains and valleys
                for k in range(section_start, min(section_end, len(x))):
                    # Position within section (0-1)
                    pos = (k - section_start) / 10
                    
                    # Create peak in the middle of the section
                    peak_factor = 1 - abs(pos - 0.5) * 2
                    
                    # Combine factors
                    height = base_height * edge_factor * (0.5 + 0.5 * peak_factor)
                    
                    # Add density-based variation
                    height += variation * 0.3 * peak_factor * edge_factor
                    
                    # Add theme-based features
                    for t in range(min(5, len(themes))):
                        if theme_strength[i, t] > 0:
                            # Create river-like features for themes
                            river_pos = (t + 1) / 6 * width - width/2  # Position the river
                            river_width = 0.5  # Width of the river
                            river_factor = max(0, 1 - abs(y[j] - river_pos) / river_width)
                            
                            # Lower terrain for rivers
                            height -= 0.2 * river_factor * theme_strength[i, t]
                    
                    Z[j, k] = height
        
        # Smooth the terrain
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=1)
        
        return X, Y, Z, themes, complexity, density


class TerrainVisualization:
    def __init__(self, master, terrain_data=None):
        self.master = master
        self.terrain_data = terrain_data
        
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create canvas for the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        if terrain_data:
            self.update_plot(terrain_data)
    
    def update_plot(self, terrain_data):
        """Update the terrain plot"""
        self.terrain_data = terrain_data
        X, Y, Z, themes, complexity, density = terrain_data
        
        # Clear previous plot
        self.ax.clear()
        # Plot the terrain surface
        terrain = self.ax.plot_surface(X, Y, Z, cmap='terrain', 
                                      linewidth=0, antialiased=True, alpha=0.8)
        
        # Add color bar
        self.fig.colorbar(terrain, ax=self.ax, shrink=0.5, aspect=5,
                          label='Cognitive Complexity')
        
        # Plot theme rivers
        for t in range(min(5, len(themes))):
            river_pos = (t + 1) / 6 * 10 - 10/2  # Position the river
            x_river = np.linspace(0, len(complexity), len(complexity) * 10)
            y_river = np.ones_like(x_river) * river_pos
            z_river = np.zeros_like(x_river)
            
            # Calculate river heights
            for i in range(len(complexity)):
                section_start = i * 10
                section_end = (i + 1) * 10
                for j in range(section_start, min(section_end, len(x_river))):
                    # Get corresponding z value from terrain
                    idx_y = np.argmin(np.abs(Y[:, 0] - river_pos))
                    z_river[j] = Z[idx_y, j] - 0.05  # Slightly below terrain
            
            # Plot the river
            self.ax.plot(x_river, y_river, z_river, color='blue', linewidth=2, alpha=0.7)
            
            # Add theme label
            if t < len(themes):
                self.ax.text(len(complexity) - 1, river_pos, Z[np.argmin(np.abs(Y[:, 0] - river_pos)), -1] + 0.1,
                             themes[t], color='black', fontsize=10)
        
        # Plot the current position
        if hasattr(self, 'current_position'):
            x, y, z = self.current_position
            self.ax.scatter([x], [y], [z], color='red', s=100, marker='o')
        
        # Set labels and title
        self.ax.set_xlabel('Book Progression')
        self.ax.set_ylabel('Conceptual Space')
        self.ax.set_zlabel('Cognitive Complexity')
        self.ax.set_title('Reading Landscape')
        
        # Adjust view angle
        self.ax.view_init(elev=30, azim=45)
        
        # Refresh canvas
        self.canvas.draw()
    
    def set_current_position(self, position):
        """Set the current reading position"""
        self.current_position = position
        if self.terrain_data:
            self.update_plot(self.terrain_data)
    
    def animate_movement(self, start_pos, end_pos, duration=1.0):
        """Animate movement from start to end position"""
        steps = 20
        x_start, y_start, z_start = start_pos
        x_end, y_end, z_end = end_pos
        
        for i in range(steps + 1):
            t = i / steps
            x = x_start + (x_end - x_start) * t
            y = y_start + (y_end - y_start) * t
            z = z_start + (z_end - z_start) * t
            
            self.set_current_position((x, y, z))
            self.master.update()
            time.sleep(duration / steps)


class MomentumEngine:
    def __init__(self):
        self.momentum = 0.0
        self.max_momentum = 10.0
        self.attention_score = 1.0
        self.last_interaction_time = time.time()
        self.reading_velocity = 0.0
        self.section_completion = 0.0
        self.current_section = 0
        self.total_sections = 0
        self.section_start_time = time.time()
        self.attention_decay_rate = 0.05  # Decay rate for attention
        self.momentum_decay_rate = 0.02  # Decay rate for momentum
        
    def update_momentum(self, interaction_intensity=1.0):
        """Update momentum based on user interaction"""
        current_time = time.time()
        time_diff = current_time - self.last_interaction_time
        
        # Decay momentum over time
        self.momentum *= max(0, 1 - self.momentum_decay_rate * time_diff)
        
        # Add new momentum from interaction
        self.momentum += interaction_intensity * self.attention_score
        
        # Cap momentum
        self.momentum = min(self.max_momentum, self.momentum)
        
        # Update last interaction time
        self.last_interaction_time = current_time
        
        # Calculate reading velocity
        self.reading_velocity = self.momentum / self.max_momentum
        
        return self.momentum, self.reading_velocity
    
    def update_attention(self, engagement_level=1.0):
        """Update attention score based on user engagement"""
        current_time = time.time()
        time_diff = current_time - self.last_interaction_time
        
        # Decay attention over time
        self.attention_score *= max(0.1, 1 - self.attention_decay_rate * time_diff)
        
        # Add new attention from engagement
        self.attention_score += engagement_level * 0.1
        
        # Cap attention score
        self.attention_score = min(1.0, max(0.1, self.attention_score))
        
        return self.attention_score
    
    def set_current_section(self, section, total_sections):
        """Set the current section being read"""
        self.current_section = section
        self.total_sections = total_sections
        self.section_completion = 0.0
        self.section_start_time = time.time()
    
    def update_section_progress(self, completion_percentage):
        """Update progress within the current section"""
        self.section_completion = completion_percentage
        
        # Calculate overall progress
        overall_progress = (self.current_section + self.section_completion) / self.total_sections
        
        return overall_progress
    
    def get_boost_opportunity(self):
        """Check if there's an opportunity for a momentum boost"""
        # Boosts are available when:
        # 1. Momentum is below threshold
        # 2. Attention is high
        # 3. Section is 25%, 50%, or 75% complete
        
        boost_available = False
        boost_type = None
        
        if self.momentum < self.max_momentum * 0.5 and self.attention_score > 0.6:
            # Check section milestones
            if abs(self.section_completion - 0.25) < 0.05:
                boost_available = True
                boost_type = "quick_question"
            elif abs(self.section_completion - 0.5) < 0.05:
                boost_available = True
                boost_type = "key_concept"
            elif abs(self.section_completion - 0.75) < 0.05:
                boost_available = True
                boost_type = "summary_challenge"
        
        return boost_available, boost_type
    
    def apply_boost(self, boost_type):
        """Apply a momentum boost"""
        if boost_type == "quick_question":
            self.momentum += self.max_momentum * 0.2
        elif boost_type == "key_concept":
            self.momentum += self.max_momentum * 0.3
        elif boost_type == "summary_challenge":
            self.momentum += self.max_momentum * 0.4
        
        # Cap momentum
        self.momentum = min(self.max_momentum, self.momentum)
        
        return self.momentum


class AdaptiveReadingPathGenerator:
    def __init__(self, complexity_profile, density_profile, theme_strength):
        self.complexity_profile = complexity_profile
        self.density_profile = density_profile
        self.theme_strength = theme_strength
        self.user_preferences = {
            "preferred_complexity": 0.5,  # 0-1 scale
            "complexity_tolerance": 0.3,  # How much deviation is acceptable
            "theme_preference": None,     # Index of preferred theme or None
            "session_duration": 30        # Preferred session length in minutes
        }
        self.reading_history = []
        
    def set_user_preferences(self, preferences):
        """Update user preferences"""
        self.user_preferences.update(preferences)
    
    def record_reading_session(self, session_data):
        """Record data from a reading session"""
        self.reading_history.append(session_data)
    
    def generate_optimal_path(self):
        """Generate optimal reading path based on user preferences and history"""
        # Calculate a "cost" for each section based on how well it matches preferences
        section_costs = np.zeros(len(self.complexity_profile))
        
        for i in range(len(self.complexity_profile)):
            # Complexity cost - how far from preferred complexity
            complexity_cost = abs(self.complexity_profile[i] - self.user_preferences["preferred_complexity"])
            complexity_cost = max(0, complexity_cost - self.user_preferences["complexity_tolerance"])
            
            # Theme preference cost
            theme_cost = 0
            if self.user_preferences["theme_preference"] is not None:
                theme_idx = self.user_preferences["theme_preference"]
                if theme_idx < self.theme_strength.shape[1]:
                    # Lower cost for sections with preferred theme
                    theme_cost = 1 - self.theme_strength[i, theme_idx]
            
            # Combine costs
            section_costs[i] = 0.7 * complexity_cost + 0.3 * theme_cost
        
        # Generate path - in this simplified version, just sort sections by cost
        section_indices = np.arange(len(section_costs))
        optimal_order = section_indices[np.argsort(section_costs)]
        
        # But we need to maintain some sequential order, so we'll divide into chunks
        # and optimize within each chunk
        chunk_size = max(1, len(optimal_order) // 5)
        chunked_order = []
        
        for i in range(0, len(optimal_order), chunk_size):
            chunk = optimal_order[i:i+chunk_size]
            # Sort chunk by original index to maintain sequence
            chunk = sorted(chunk)
            chunked_order.extend(chunk)
        
        return chunked_order
    
    def identify_focus_zones(self):
        """Identify sections requiring heightened focus"""
        focus_zones = []
        
        for i in range(len(self.complexity_profile)):
            # Mark as focus zone if complexity is high
            if self.complexity_profile[i] > 0.7:
                focus_zones.append(i)
            # Or if density is high
            elif self.density_profile[i] > 0.8:
                focus_zones.append(i)
        
        return focus_zones
    
    def identify_rest_points(self):
        """Identify optimal rest points"""
        rest_points = []
        
        for i in range(1, len(self.complexity_profile) - 1):
            # Good rest point if it's after a high complexity section
            # and before another high complexity section
            if (self.complexity_profile[i-1] > 0.7 and 
                self.complexity_profile[i] < 0.5 and
                self.complexity_profile[i+1] > 0.6):
                rest_points.append(i)
        
        # Ensure sufficient rest points
        min_rest_points = max(1, len(self.complexity_profile) // 5)
        if len(rest_points) < min_rest_points:
            # Add more rest points at regular intervals
            interval = len(self.complexity_profile) // (min_rest_points + 1)
            for i in range(interval, len(self.complexity_profile), interval):
                if i not in rest_points:
                    rest_points.append(i)
        
        return sorted(rest_points)


class KnowledgeCrystallizationEngine:
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model if nlp_model else spacy.load('en_core_web_sm')
        self.knowledge_vault = []
        self.concept_map = {}
        self.mastery_levels = {}
    
    def extract_key_concepts(self, text):
        """Extract key concepts from text"""
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract key noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks 
                        if len(chunk.text.split()) > 1]
        
        # Extract potentially important sentences
        important_sentences = []
        for sent in doc.sents:
            # Check for definition patterns
            if any(token.lemma_ in ["be", "define", "refer", "mean"] for token in sent):
                important_sentences.append(sent.text)
            
            # Check for key phrases
            if any(phrase in sent.text for phrase in ["key", "important", "significant", "critical"]):
                important_sentences.append(sent.text)
        
        return {
            "entities": entities,
            "noun_phrases": noun_phrases,
            "important_sentences": important_sentences
        }
    
    def crystallize_concept(self, text, section_index):
        """Create a knowledge crystal from text"""
        concepts = self.extract_key_concepts(text)
        
        # Create a crystal object
        crystal = {
            "section": section_index,
            "entities": concepts["entities"],
            "key_phrases": concepts["noun_phrases"][:5],  # Limit to top 5
            "definitions": concepts["important_sentences"][:3],  # Limit to top 3
            "mastery_level": 0.0  # Initial mastery level
        }
        
        # Add to knowledge vault
        self.knowledge_vault.append(crystal)
        
        # Update concept map
        for entity, _ in crystal["entities"]:
            if entity not in self.concept_map:
                self.concept_map[entity] = []
            self.concept_map[entity].append(len(self.knowledge_vault) - 1)  # Index of this crystal
        
        return crystal
    
    def update_mastery(self, crystal_index, comprehension_score):
        """Update mastery level for a crystal"""
        if 0 <= crystal_index < len(self.knowledge_vault):
            # Update mastery with exponential moving average
            current = self.knowledge_vault[crystal_index]["mastery_level"]
            self.knowledge_vault[crystal_index]["mastery_level"] = 0.7 * current + 0.3 * comprehension_score
            
            # Update mastery for related concepts
            for entity, _ in self.knowledge_vault[crystal_index]["entities"]:
                if entity in self.mastery_levels:
                    self.mastery_levels[entity] = 0.8 * self.mastery_levels[entity] + 0.2 * comprehension_score
                else:
                    self.mastery_levels[entity] = comprehension_score
    
    def get_mastery_visualization(self):
        """Generate data for mastery visualization"""
        if not self.knowledge_vault:
            return None
        
        # Group by entity type
        entity_groups = {}
        for crystal in self.knowledge_vault:
            for entity, entity_type in crystal["entities"]:
                if entity_type not in entity_groups:
                    entity_groups[entity_type] = []
                
                # Add if not already present
                if entity not in [e[0] for e in entity_groups[entity_type]]:
                    mastery = self.mastery_levels.get(entity, 0.0)
                    entity_groups[entity_type].append((entity, mastery))
        
        return entity_groups
    
    def get_related_concepts(self, concept_name):
        """Get concepts related to the given concept"""
        related = {}
        
        if concept_name in self.concept_map:
            crystal_indices = self.concept_map[concept_name]
            for idx in crystal_indices:
                crystal = self.knowledge_vault[idx]
                for entity, entity_type in crystal["entities"]:
                    if entity != concept_name:
                        if entity not in related:
                            related[entity] = {"type": entity_type, "strength": 0}
                        related[entity]["strength"] += 1
        
        # Sort by strength
        return sorted(related.items(), key=lambda x: x[1]["strength"], reverse=True)


class TemporalEngagementSystem:
    def __init__(self):
        self.session_start_time = None
        self.attention_span_estimate = 15 * 60  # Default 15 minutes in seconds
        self.optimal_chunk_size = 5 * 60  # Default 5 minutes in seconds
        self.flow_state_history = []  # Track flow state over time
        self.break_times = []  # Track when breaks were taken
        self.attention_metrics = {
            "sustained_focus": [],      # Duration able to maintain focus
            "engagement_scores": [],    # Engagement scores over time
            "distraction_points": []    # Times when attention was broken
        }
        
    def start_session(self):
        """Start a new reading session"""
        self.session_start_time = time.time()
        self.flow_state_history = []
        
    def update_flow_state(self, reading_speed, comprehension_score):
        """Update flow state based on reading metrics"""
        current_time = time.time()
        
        if self.session_start_time is None:
            self.start_session()
        
        session_duration = current_time - self.session_start_time
        
        # Calculate flow score (0-1)
        # High reading speed + high comprehension = flow state
        flow_score = (reading_speed * 0.5) + (comprehension_score * 0.5)
        
        self.flow_state_history.append((session_duration, flow_score))
        
        # Return whether in flow state
        return flow_score > 0.7
    
    def register_distraction(self):
        """Register a point where the user was distracted"""
        current_time = time.time()
        
        if self.session_start_time is None:
            self.start_session()
        
        session_duration = current_time - self.session_start_time
        
        # Add to distraction points
        self.attention_metrics["distraction_points"].append(session_duration)
        
        # Update sustained focus if there are at least two distraction points
        if len(self.attention_metrics["distraction_points"]) > 1:
            last_distraction = self.attention_metrics["distraction_points"][-2]
            focus_duration = session_duration - last_distraction
            self.attention_metrics["sustained_focus"].append(focus_duration)
            
            # Update attention span estimate
            if len(self.attention_metrics["sustained_focus"]) >= 3:
                # Use the average of the last 3 sustained focus periods
                recent_focus = self.attention_metrics["sustained_focus"][-3:]
                self.attention_span_estimate = sum(recent_focus) / len(recent_focus)
                
                # Update optimal chunk size - typically 70-80% of attention span
                self.optimal_chunk_size = self.attention_span_estimate * 0.75
    
    def should_take_break(self, current_complexity):
        """Determine if it's a good time for a break"""
        current_time = time.time()
        
        if self.session_start_time is None:
            return False
        
        session_duration = current_time - self.session_start_time
        
        # Check if we've been reading for longer than attention span
        time_factor = session_duration > self.attention_span_estimate
        
        # Check if complexity is low (good breaking point)
        complexity_factor = current_complexity < 0.3
        
        # Check if we're not in flow state
        flow_factor = False
        if self.flow_state_history:
            recent_flow = [f[1] for f in self.flow_state_history[-3:]]
            flow_factor = sum(recent_flow) / len(recent_flow) < 0.5
        
        # Combine factors
        should_break = time_factor and (complexity_factor or flow_factor)
        
        if should_break:
            # Record break time
            self.break_times.append(session_duration)
            # Reset session timer
            self.session_start_time = current_time
        
        return should_break
    
    def calibrate_with_feedback(self, focus_rating):
        """Calibrate the system with explicit user feedback"""
        # focus_rating is 1-5 where 5 is highly focused
        normalized_rating = focus_rating / 5.0
        
        # Update engagement scores
        current_time = time.time()
        if self.session_start_time is None:
            self.start_session()
        
        session_duration = current_time - self.session_start_time
        self.attention_metrics["engagement_scores"].append((session_duration, normalized_rating))
        
        # If rating is low, treat as distraction point
        if normalized_rating < 0.4:
            self.register_distraction()
        
        # Adjust attention span based on rating
        adjustment_factor = (normalized_rating - 0.5) * 0.2  # -10% to +10%
        self.attention_span_estimate *= (1 + adjustment_factor)
        
        # Update optimal chunk size
        self.optimal_chunk_size = self.attention_span_estimate * 0.75


class MomentumReadingApp:
    def __init__(self, root):
        """Initialize the Momentum Reading App"""
        self.root = root
        self.root.title("Momentum Reading")
        self.root.geometry("1200x800")
        
        # Set up the main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize engines and data
        self.content_analyzer = ContentAnalyzer()
        self.momentum_engine = MomentumEngine()
        self.temporal_system = TemporalEngagementSystem()
        self.knowledge_engine = None
        self.path_generator = None
        self.book_text = ""
        self.book_sections = []
        self.current_section_index = 0
        self.terrain_data = None
        self.landscape_window = None  # To track the dialog box
        self.terrain_viz = None       # To track the terrain visualization instance
        
        # Create UI components
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Top panel for controls
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Load book button
        self.load_btn = ttk.Button(self.control_frame, text="Load Book", command=self.load_book)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Button to show reading landscape
        self.show_landscape_btn = ttk.Button(self.control_frame, text="Show Reading Landscape", command=self.show_landscape)
        self.show_landscape_btn.pack(side=tk.LEFT, padx=5)
        
        # Book info label
        self.book_info = ttk.Label(self.control_frame, text="No book loaded")
        self.book_info.pack(side=tk.LEFT, padx=20)
        
        # Progress bar
        self.progress_frame = ttk.Frame(self.control_frame)
        self.progress_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, 
                                            length=300, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X, expand=True)
        
        # Middle panel for reading content
        self.middle_frame = ttk.Frame(self.main_frame)
        self.middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Reading content frame (occupies the entire middle frame)
        self.reading_frame = ttk.LabelFrame(self.middle_frame, text="Reading Content")
        self.reading_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text widget for content
        self.content_text = tk.Text(self.reading_frame, wrap=tk.WORD, 
                                    font=("Georgia", 12), padx=10, pady=10)
        self.scrollbar = ttk.Scrollbar(self.reading_frame, orient=tk.VERTICAL, 
                                       command=self.content_text.yview)
        self.content_text.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.content_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom panel for momentum visualization and controls
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Momentum meter
        self.momentum_frame = ttk.LabelFrame(self.bottom_frame, text="Reading Momentum")
        self.momentum_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.momentum_var = tk.DoubleVar(value=0)
        self.momentum_meter = ttk.Progressbar(self.momentum_frame, orient=tk.HORIZONTAL, 
                                              length=200, mode='determinate', variable=self.momentum_var)
        self.momentum_meter.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        # Navigation buttons
        self.nav_frame = ttk.Frame(self.bottom_frame)
        self.nav_frame.pack(side=tk.RIGHT, padx=5)
        self.prev_btn = ttk.Button(self.nav_frame, text="Previous", command=self.previous_section)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn = ttk.Button(self.nav_frame, text="Next", command=self.next_section)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Focus rating frame
        self.focus_frame = ttk.LabelFrame(self.bottom_frame, text="Focus Rating")
        self.focus_frame.pack(side=tk.RIGHT, padx=20)
        self.focus_var = tk.IntVar(value=3)
        self.focus_scale = ttk.Scale(self.focus_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                                     variable=self.focus_var, command=self.focus_changed)
        self.focus_scale.pack(padx=10, pady=5)
        focus_labels = ttk.Frame(self.focus_frame)
        focus_labels.pack(fill=tk.X, padx=5)
        ttk.Label(focus_labels, text="Low").pack(side=tk.LEFT)
        ttk.Label(focus_labels, text="High").pack(side=tk.RIGHT)
        
        # Setup periodic updates
        self.root.after(1000, self.periodic_update)

    def show_landscape(self):
        """Show the reading landscape in a dialog box"""
        if self.terrain_data is None:
            messagebox.showinfo("Info", "Please load a book first.")
            return
        # If the dialog is already open, bring it to the front
        if self.landscape_window is not None and self.landscape_window.winfo_exists():
            self.landscape_window.lift()
        else:
            # Create a new dialog box
            self.landscape_window = tk.Toplevel(self.root)
            self.landscape_window.title("Reading Landscape")
            self.terrain_viz = TerrainVisualization(self.landscape_window, self.terrain_data)
            # Handle window close event
            self.landscape_window.protocol("WM_DELETE_WINDOW", self.on_landscape_close)

    def on_landscape_close(self):
        """Handle closing of the landscape dialog box"""
        self.landscape_window.destroy()
        self.landscape_window = None
        self.terrain_viz = None

    def load_book(self):
        """Load a book file"""
        file_path = filedialog.askopenfilename(
            title="Select a book file",
            filetypes=[("PDF files", "*.pdf"), ("EPUB files", "*.epub")]
        )
        if not file_path:
            return
        try:
            self.book_text = self.content_analyzer.extract_text(file_path)
            if not self.book_text.strip():
                messagebox.showerror("Error", "No text could be extracted from the book.")
                return
            filename = os.path.basename(file_path)
            self.book_info.config(text=f"Book: {filename}")
            self.process_book()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load book: {str(e)}")

    def process_book(self):
        """Process the loaded book"""
        # Split into sections
        self.book_sections = self.content_analyzer.split_into_sections(self.book_text)
        if not self.book_sections:
            self.book_sections = [self.book_text]
        
        # Generate terrain data
        self.terrain_data = self.content_analyzer.generate_reading_landscape(self.book_text)
        
        # Update terrain visualization if the dialog is open
        if self.terrain_viz and self.terrain_viz.master.winfo_exists():
            self.terrain_viz.update_plot(self.terrain_data)
        
        # Set up engines
        self.momentum_engine.set_current_section(0, len(self.book_sections))
        self.knowledge_engine = KnowledgeCrystallizationEngine(nlp)
        _, _, theme_strength, _ = self.content_analyzer.generate_terrain_data(self.book_sections)
        self.path_generator = AdaptiveReadingPathGenerator(
            self.terrain_data[4], self.terrain_data[5], theme_strength
        )
        self.temporal_system.start_session()
        
        # Load first section
        self.current_section_index = 0
        self.load_section(self.current_section_index)

    def load_section(self, section_index):
        """Load a specific section into the reader"""
        if 0 <= section_index < len(self.book_sections):
            self.current_section_index = section_index
            self.content_text.delete(1.0, tk.END)
            self.content_text.insert(tk.END, self.book_sections[section_index])
            self.momentum_engine.set_current_section(section_index, len(self.book_sections))
            progress = (section_index + 1) / len(self.book_sections)
            self.progress_var.set(progress * 100)
            
            # Update terrain position if the dialog is open
            if self.terrain_viz and self.terrain_viz.master.winfo_exists():
                complexity_profile = self.terrain_data[4]
                x_pos = section_index
                y_pos = 0
                z_pos = complexity_profile[section_index] if section_index < len(complexity_profile) else 0
                if hasattr(self.terrain_viz, 'current_position'):
                    start_pos = self.terrain_viz.current_position
                    end_pos = (x_pos, y_pos, z_pos)
                    self.terrain_viz.animate_movement(start_pos, end_pos)
                else:
                    self.terrain_viz.set_current_position((x_pos, y_pos, z_pos))
            
            # Knowledge crystallization and focus zones
            if self.knowledge_engine:
                self.knowledge_engine.crystallize_concept(self.book_sections[section_index], section_index)
            if self.path_generator:
                focus_zones = self.path_generator.identify_focus_zones()
                self.content_text.configure(background="#FFFFE0" if section_index in focus_zones else "white")

    def next_section(self):
        """Move to the next section"""
        if self.current_section_index < len(self.book_sections) - 1:
            # Check if we should take a break
            if self.temporal_system and self.terrain_data:
                complexity = self.terrain_data[4][self.current_section_index]
                if self.temporal_system.should_take_break(complexity):
                    self.offer_break()
            
            # Update momentum
            self.momentum_engine.update_momentum(1.0)
            self.momentum_var.set(self.momentum_engine.momentum * 10)
            
            # Load next section
            self.load_section(self.current_section_index + 1)
            
            # Update flow state
            if self.temporal_system:
                self.temporal_system.update_flow_state(
                    self.momentum_engine.reading_velocity,
                    0.8  # Assuming good comprehension for now
                )

    def previous_section(self):
        """Move to the previous section"""
        if self.current_section_index > 0:
            # Update momentum (less than going forward)
            self.momentum_engine.update_momentum(0.5)
            self.momentum_var.set(self.momentum_engine.momentum * 10)
            
            # Load previous section
            self.load_section(self.current_section_index - 1)

    def focus_changed(self, event=None):
        """Handle focus rating changes"""
        focus_value = self.focus_var.get()
        
        # Update temporal system
        if self.temporal_system:
            self.temporal_system.calibrate_with_feedback(focus_value)
        
        # Update attention score in momentum engine
        if self.momentum_engine:
            engagement_level = focus_value / 5.0
            self.momentum_engine.update_attention(engagement_level)

    def offer_break(self):
        """Offer a break to the user"""
        take_break = messagebox.askyesno(
            "Break Time",
            "Your attention metrics suggest it's a good time for a short break. Take a 2-minute break?"
        )
        if take_break:
            self.start_break_timer()

    def start_break_timer(self):
        """Start a timer for the break"""
        break_window = tk.Toplevel(self.root)
        break_window.title("Break Time")
        break_window.geometry("300x200")
        ttk.Label(break_window, text="Take a short break", font=("Arial", 14)).pack(pady=20)
        time_var = tk.StringVar(value="2:00")
        time_label = ttk.Label(break_window, textvariable=time_var, font=("Arial", 24))
        time_label.pack(pady=20)
        
        def countdown(seconds_left):
            if seconds_left <= 0:
                break_window.destroy()
                return
            minutes = seconds_left // 60
            seconds = seconds_left % 60
            time_var.set(f"{minutes}:{seconds:02d}")
            break_window.after(1000, countdown, seconds_left - 1)
        
        countdown(120)
        break_window.update_idletasks()
        width = break_window.winfo_width()
        height = break_window.winfo_height()
        x = (break_window.winfo_screenwidth() // 2) - (width // 2)
        y = (break_window.winfo_screenheight() // 2) - (height // 2)
        break_window.geometry(f"{width}x{height}+{x}+{y}")
        break_window.focus_set()

    def periodic_update(self):
        """Periodic updates"""
        if self.momentum_engine:
            self.momentum_engine.update_momentum(0)
            self.momentum_var.set(self.momentum_engine.momentum * 10)
            boost_available, boost_type = self.momentum_engine.get_boost_opportunity()
            if boost_available:
                self.offer_boost(boost_type)
        self.root.after(1000, self.periodic_update)

    def offer_boost(self, boost_type):
        """Offer a momentum boost to the user"""
        if boost_type == "quick_question":
            self.show_quick_question()
        elif boost_type == "key_concept":
            self.show_key_concept()
        elif boost_type == "summary_challenge":
            self.show_summary_challenge()

    def show_quick_question(self):
        """Show a quick question to boost momentum"""
        section_text = self.book_sections[self.current_section_index]
        doc = nlp(section_text[:1000])
        statements = [sent.text for sent in doc.sents if 20 < len(sent.text) < 100 and any(token.pos_ == "VERB" for token in sent)]
        if statements:
            import random
            statement = random.choice(statements)
            question = "True or False: " + statement
            answer = messagebox.askyesno("Quick Check", question)
            self.momentum_engine.apply_boost("quick_question")
            self.momentum_var.set(self.momentum_engine.momentum * 10)

    def show_key_concept(self):
        """Show a key concept to boost momentum"""
        if self.knowledge_engine and self.knowledge_engine.knowledge_vault:
            crystal = self.knowledge_engine.knowledge_vault[-1]
            concept_window = tk.Toplevel(self.root)
            concept_window.title("Key Concept")
            concept_window.geometry("400x300")
            ttk.Label(concept_window, text="Key Concept", font=("Arial", 16)).pack(pady=10)
            if crystal["key_phrases"]:
                phrase_frame = ttk.LabelFrame(concept_window, text="Key Phrases")
                phrase_frame.pack(fill=tk.X, padx=20, pady=10)
                for phrase in crystal["key_phrases"][:3]:
                    ttk.Label(phrase_frame, text="• " + phrase).pack(anchor=tk.W, padx=10, pady=2)
            if crystal["definitions"]:
                def_frame = ttk.LabelFrame(concept_window, text="Definitions")
                def_frame.pack(fill=tk.X, padx=20, pady=10)
                for definition in crystal["definitions"][:2]:
                    ttk.Label(def_frame, text=definition, wraplength=350).pack(padx=10, pady=5)
            ttk.Button(concept_window, text="Got it!", command=lambda: self.concept_understood(concept_window)).pack(pady=20)

    def concept_understood(self, window):
        """Handle concept understood button click"""
        window.destroy()
        self.momentum_engine.apply_boost("key_concept")
        self.momentum_var.set(self.momentum_engine.momentum * 10)
        if self.knowledge_engine and self.knowledge_engine.knowledge_vault:
            self.knowledge_engine.update_mastery(len(self.knowledge_engine.knowledge_vault) - 1, 0.8)

    def show_summary_challenge(self):
        """Show a summary challenge to boost momentum"""
        challenge_window = tk.Toplevel(self.root)
        challenge_window.title("Summary Challenge")
        challenge_window.geometry("500x400")
        ttk.Label(challenge_window, text="Summarize what you've learned so far", font=("Arial", 16)).pack(pady=10)
        ttk.Label(challenge_window, text="Write a brief summary of the key points from this section", wraplength=450).pack(pady=10)
        summary_text = tk.Text(challenge_window, height=10, width=50)
        summary_text.pack(padx=20, pady=10)
        ttk.Button(challenge_window, text="Submit", command=lambda: self.process_summary(challenge_window, summary_text.get(1.0, tk.END))).pack(pady=10)

    def process_summary(self, window, summary):
        """Process the submitted summary"""
        self.momentum_engine.apply_boost("summary_challenge")
        self.momentum_var.set(self.momentum_engine.momentum * 10)
        window.destroy()
        if self.temporal_system:
            self.temporal_system.update_flow_state(self.momentum_engine.reading_velocity, 0.9)


if __name__ == "__main__":
    root = tk.Tk()
    app = MomentumReadingApp(root)
    root.mainloop()