import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font
import tkinter.scrolledtext as scrolledtext

class MinimalLang:
    """Definition of the Minimal Programming Language"""
    
    # Language keywords
    KEYWORDS = [
        "func", "var", "if", "else", "loop", "return", 
        "true", "false", "null", "break", "continue",
        "and", "or", "not", "import", "export"
    ]
    
    # Operators and punctuation
    OPERATORS = [
        "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", 
        "<=", ">=", ".", ",", ":", ";", "(", ")", "{", "}", 
        "[", "]", "->", "=>", "<<", ">>"
    ]
    
    # Data types
    TYPES = ["num", "str", "bool", "list", "map"]
    
    # Comment symbol
    COMMENT = "#"
    
    # String delimiters
    STRING_DELIMITERS = ['"', "'"]
    
    @staticmethod
    def get_autocomplete_list():
        """Get full list of language elements for autocomplete"""
        return MinimalLang.KEYWORDS + MinimalLang.OPERATORS + MinimalLang.TYPES
    
    @staticmethod
    def tokenize(text):
        """Simple tokenizer for syntax highlighting"""
        tokens = []
        i = 0
        while i < len(text):
            char = text[i]
            
            # Handle whitespace
            if char.isspace():
                i += 1
                continue
                
            # Handle comments
            if char == MinimalLang.COMMENT:
                start = i
                while i < len(text) and text[i] != '\n':
                    i += 1
                tokens.append(('comment', text[start:i]))
                continue
                
            # Handle strings
            if char in MinimalLang.STRING_DELIMITERS:
                delimiter = char
                start = i
                i += 1
                while i < len(text) and text[i] != delimiter:
                    if text[i] == '\\' and i + 1 < len(text):
                        i += 2
                    else:
                        i += 1
                if i < len(text):  # Include closing delimiter
                    i += 1
                tokens.append(('string', text[start:i]))
                continue
                
            # Handle numbers
            if char.isdigit():
                start = i
                while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                    i += 1
                tokens.append(('number', text[start:i]))
                continue
                
            # Handle identifiers and keywords
            if char.isalpha() or char == '_':
                start = i
                while i < len(text) and (text[i].isalnum() or text[i] == '_'):
                    i += 1
                word = text[start:i]
                if word in MinimalLang.KEYWORDS:
                    tokens.append(('keyword', word))
                elif word in MinimalLang.TYPES:
                    tokens.append(('type', word))
                else:
                    tokens.append(('identifier', word))
                continue
                
            # Handle operators
            for op in sorted(MinimalLang.OPERATORS, key=len, reverse=True):
                if text[i:i+len(op)] == op:
                    tokens.append(('operator', op))
                    i += len(op)
                    break
            else:
                # If no operator matches, treat it as a symbol
                tokens.append(('symbol', char))
                i += 1
                
        return tokens


class CodeEditor:
    """Minimal Programming Language Code Editor"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Minimal Language Editor")
        self.root.geometry("1000x700")
        
        # Current file
        self.current_file = None
        
        # Setup UI
        self._setup_ui()
        
        # Configure syntax highlighting
        self._setup_syntax_highlighting()
        
        # Configure autocomplete
        self._setup_autocomplete()
        
        # Set up key bindings
        self._setup_key_bindings()
        
        # Status message
        self.status_var = tk.StringVar()
        self.status_var.set("New File")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _setup_ui(self):
        """Set up the user interface"""
        # Menu bar
        self._setup_menu()
        
        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Line numbers and text area in a horizontal layout
        self.text_frame = tk.Frame(self.main_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Line numbers
        self.line_numbers = tk.Canvas(self.text_frame, width=30, bg='#f0f0f0')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Text editor with syntax highlighting
        self.text_area = scrolledtext.ScrolledText(
            self.text_frame, 
            wrap=tk.WORD, 
            undo=True,
            bg='#FFFFFF',
            fg='#000000',
            insertbackground='#000000',
            font=('Courier New', 12),
            selectbackground='#ADD6FF'
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Terminal output area
        self.terminal_frame = tk.Frame(self.main_frame, height=150)
        self.terminal_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(self.terminal_frame, text="Terminal Output").pack(anchor=tk.W)
        
        self.terminal = scrolledtext.ScrolledText(
            self.terminal_frame,
            wrap=tk.WORD,
            bg='#1E1E1E',
            fg='#FFFFFF',
            font=('Courier New', 10),
            height=10
        )
        self.terminal.pack(fill=tk.BOTH, expand=True)
        
        # Make sure the text area has focus
        self.text_area.focus_set()
        
    def _setup_menu(self):
        """Set up the menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_as_file, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=lambda: self.text_area.event_generate("<<Undo>>"), accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=lambda: self.text_area.event_generate("<<Redo>>"), accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", command=lambda: self.text_area.event_generate("<<Cut>>"), accelerator="Ctrl+X")
        edit_menu.add_command(label="Copy", command=lambda: self.text_area.event_generate("<<Copy>>"), accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=lambda: self.text_area.event_generate("<<Paste>>"), accelerator="Ctrl+V")
        edit_menu.add_separator()
        edit_menu.add_command(label="Find", command=self.show_find_dialog, accelerator="Ctrl+F")
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # Run menu
        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="Run Program", command=self.run_code, accelerator="F5")
        menubar.add_cascade(label="Run", menu=run_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Language Reference", command=self.show_language_reference)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
    def _setup_syntax_highlighting(self):
        """Set up syntax highlighting tags"""
        self.text_area.tag_configure("keyword", foreground="#CC7832")  # Orange-red for keywords
        self.text_area.tag_configure("operator", foreground="#A9B7C6")  # Light gray for operators
        self.text_area.tag_configure("string", foreground="#6A8759")    # Green for strings
        self.text_area.tag_configure("number", foreground="#6897BB")    # Blue for numbers
        self.text_area.tag_configure("comment", foreground="#808080")   # Gray for comments
        self.text_area.tag_configure("type", foreground="#B5B6E3")      # Light purple for types
        self.text_area.tag_configure("identifier", foreground="#A9B7C6") # Light gray for identifiers
        
        # Bind events for syntax highlighting
        self.text_area.bind("<KeyRelease>", self.highlight_syntax)
        
    def _setup_autocomplete(self):
        """Set up autocomplete functionality"""
        self.autocomplete_list = MinimalLang.get_autocomplete_list()
        self.autocomplete_window = None
        self.autocomplete_listbox = None
        
        # Bind events for autocomplete
        self.text_area.bind("<Control-space>", self.show_autocomplete)
        self.text_area.bind("<Tab>", self.handle_tab)
        
    def _setup_key_bindings(self):
        """Set up key bindings for the editor"""
        self.text_area.bind("<Control-n>", lambda e: self.new_file())
        self.text_area.bind("<Control-o>", lambda e: self.open_file())
        self.text_area.bind("<Control-s>", lambda e: self.save_file())
        self.text_area.bind("<Control-Shift-S>", lambda e: self.save_as_file())
        self.text_area.bind("<F5>", lambda e: self.run_code())
        self.text_area.bind("<Control-f>", lambda e: self.show_find_dialog())
        
        # Bind for updating line numbers
        self.text_area.bind("<KeyRelease>", self.update_line_numbers)
        self.text_area.bind("<MouseWheel>", self.update_line_numbers)
        self.text_area.bind("<Button-4>", self.update_line_numbers)
        self.text_area.bind("<Button-5>", self.update_line_numbers)
        
    def update_line_numbers(self, event=None):
        """Update the line numbers display"""
        # Clear the canvas
        self.line_numbers.delete("all")
        
        # Get font properties
        font_family = self.text_area.cget("font")
        font_size = font.Font(font=font_family).actual()["size"]
        
        # Get the first and last visible line
        first_line = self.text_area.index("@0,0").split('.')[0]
        last_line = self.text_area.index(f"@0,{self.text_area.winfo_height()}").split('.')[0]
        
        # Draw line numbers
        for i in range(int(first_line), int(last_line) + 1):
            y_coord = self.text_area.dlineinfo(f"{i}.0")
            if y_coord:
                self.line_numbers.create_text(15, y_coord[1], text=i, anchor="center", font=("Courier New", font_size))
                
        self.highlight_syntax()
    
    def highlight_syntax(self, event=None):
        """Apply syntax highlighting to the current text"""
        # Remove all existing tags
        for tag in ["keyword", "operator", "string", "number", "comment", "type", "identifier"]:
            self.text_area.tag_remove(tag, "1.0", "end")
        
        # Get the current text
        content = self.text_area.get("1.0", "end-1c")
        
        # Tokenize the content
        tokens = MinimalLang.tokenize(content)
        
        # Track position in the text
        pos = 0
        
        # Apply tags based on token types
        for token_type, token_text in tokens:
            start_index = "1.0+" + str(pos) + "c"
            end_index = "1.0+" + str(pos + len(token_text)) + "c"
            self.text_area.tag_add(token_type, start_index, end_index)
            pos += len(token_text)
    
    def show_autocomplete(self, event=None):
        """Show the autocomplete dropdown"""
        # Close existing autocomplete window if open
        self.close_autocomplete()
        
        # Get current word
        current_index = self.text_area.index(tk.INSERT)
        line, col = map(int, current_index.split('.'))
        line_text = self.text_area.get(f"{line}.0", f"{line}.{col}")
        
        # Find the start of the current word
        word_start = col
        for i in range(col-1, -1, -1):
            if i < 0 or not (line_text[i].isalnum() or line_text[i] == '_'):
                break
            word_start = i
        
        current_word = line_text[word_start:col]
        
        # Filter autocomplete options based on current word
        if current_word:
            filtered_options = [opt for opt in self.autocomplete_list if opt.startswith(current_word)]
            
            if filtered_options:
                # Create a new Toplevel window for autocomplete
                self.autocomplete_window = tk.Toplevel(self.root)
                self.autocomplete_window.overrideredirect(True)
                
                # Position the window near the cursor
                x, y, _, height = self.text_area.bbox(current_index)
                x = x + self.text_area.winfo_rootx()
                y = y + height + self.text_area.winfo_rooty()
                self.autocomplete_window.geometry(f"+{x}+{y}")
                
                # Create a listbox for suggestions
                self.autocomplete_listbox = tk.Listbox(self.autocomplete_window, height=min(10, len(filtered_options)))
                self.autocomplete_listbox.pack(fill=tk.BOTH, expand=True)
                
                # Add options to listbox
                for option in filtered_options:
                    self.autocomplete_listbox.insert(tk.END, option)
                
                # Bind events for selection
                self.autocomplete_listbox.bind("<ButtonRelease-1>", self.select_autocomplete)
                self.autocomplete_listbox.bind("<Return>", self.select_autocomplete)
                self.autocomplete_listbox.bind("<Escape>", self.close_autocomplete)
                
                # Select first option
                self.autocomplete_listbox.selection_set(0)
                self.autocomplete_listbox.focus_set()
                
                # Save current word info for completion
                self.autocomplete_start = f"{line}.{word_start}"
                self.autocomplete_end = current_index
        
        return "break"
    
    def close_autocomplete(self, event=None):
        """Close the autocomplete window"""
        if self.autocomplete_window:
            self.autocomplete_window.destroy()
            self.autocomplete_window = None
            self.autocomplete_listbox = None
        
    def select_autocomplete(self, event=None):
        """Select an item from the autocomplete list"""
        if self.autocomplete_listbox:
            # Get selected option
            selected_index = self.autocomplete_listbox.curselection()
            if selected_index:
                selected_option = self.autocomplete_listbox.get(selected_index)
                
                # Replace current word with selected option
                self.text_area.delete(self.autocomplete_start, self.autocomplete_end)
                self.text_area.insert(self.autocomplete_start, selected_option)
                
                # Close autocomplete window
                self.close_autocomplete()
                
                # Give focus back to text area
                self.text_area.focus_set()
        
        return "break"
    
    def handle_tab(self, event=None):
        """Handle tab key press - for indentation or autocomplete selection"""
        if self.autocomplete_window:
            return self.select_autocomplete()
        else:
            # Insert 4 spaces for indentation
            self.text_area.insert(tk.INSERT, "    ")
            return "break"
    
    def new_file(self):
        """Create a new file"""
        # Check if current file needs saving
        if self.text_area.edit_modified():
            save = messagebox.askyesnocancel("Save Changes", "Do you want to save changes?")
            if save:
                self.save_file()
            elif save is None:  # Cancel was pressed
                return
        
        # Clear text area
        self.text_area.delete("1.0", tk.END)
        self.text_area.edit_modified(False)
        self.current_file = None
        self.status_var.set("New File")
        self.root.title("Minimal Language Editor")
    
    def open_file(self):
        """Open a file"""
        # Check if current file needs saving
        if self.text_area.edit_modified():
            save = messagebox.askyesnocancel("Save Changes", "Do you want to save changes?")
            if save:
                self.save_file()
            elif save is None:  # Cancel was pressed
                return
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            filetypes=[("Minimal Files", "*.min"), ("All Files", "*.*")]
        )
        
        if file_path:
            # Read file content
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                
                # Update text area
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert("1.0", content)
                self.text_area.edit_modified(False)
                
                # Update current file info
                self.current_file = file_path
                self.status_var.set(f"Opened: {os.path.basename(file_path)}")
                self.root.title(f"{os.path.basename(file_path)} - Minimal Language Editor")
                
                # Apply syntax highlighting
                self.highlight_syntax()
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")
    
    def save_file(self):
        """Save the current file"""
        if self.current_file:
            try:
                content = self.text_area.get("1.0", "end-1c")
                with open(self.current_file, 'w') as file:
                    file.write(content)
                self.text_area.edit_modified(False)
                self.status_var.set(f"Saved: {os.path.basename(self.current_file)}")
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
                return False
        else:
            return self.save_as_file()
    
    def save_as_file(self):
        """Save the file with a new name"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".min",
            filetypes=[("Minimal Files", "*.min"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.current_file = file_path
            self.root.title(f"{os.path.basename(file_path)} - Minimal Language Editor")
            return self.save_file()
        
        return False
    
    def run_code(self):
        """Run the current code (placeholder for actual interpreter)"""
        # Save before running
        if self.text_area.edit_modified():
            save = messagebox.askyesno("Save Changes", "Save changes before running?")
            if save:
                saved = self.save_file()
                if not saved:
                    return
        
        # Clear terminal
        self.terminal.delete("1.0", tk.END)
        
        # Get code
        code = self.text_area.get("1.0", "end-1c")
        
        # For now, just display a message - in a real implementation this would call the interpreter
        self.terminal.insert("1.0", "=== Program Output ===\n")
        self.terminal.insert("end", "Running Minimal language code...\n")
        self.terminal.insert("end", "This is a placeholder for the actual interpreter.\n")
        self.terminal.insert("end", f"Code length: {len(code)} characters\n")
        self.terminal.insert("end", "=== Program Ended ===\n")
    
    def show_find_dialog(self):
        """Show the find dialog"""
        find_dialog = tk.Toplevel(self.root)
        find_dialog.title("Find")
        find_dialog.geometry("300x80")
        find_dialog.transient(self.root)
        find_dialog.resizable(False, False)
        
        # Find entry
        tk.Label(find_dialog, text="Find:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        find_entry = tk.Entry(find_dialog, width=30)
        find_entry.grid(row=0, column=1, padx=5, pady=5)
        find_entry.focus_set()
        
        # Case sensitivity
        case_var = tk.BooleanVar()
        case_check = tk.Checkbutton(find_dialog, text="Match case", variable=case_var)
        case_check.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        
        # Find button
        def find_text():
            text = find_entry.get()
            if text:
                # Clear previous highlights
                self.text_area.tag_remove("search", "1.0", "end")
                
                start_pos = "1.0"
                while True:
                    if case_var.get():
                        start_pos = self.text_area.search(text, start_pos, stopindex="end", nocase=False)
                    else:
                        start_pos = self.text_area.search(text, start_pos, stopindex="end", nocase=True)
                    
                    if not start_pos:
                        break
                    
                    end_pos = f"{start_pos}+{len(text)}c"
                    self.text_area.tag_add("search", start_pos, end_pos)
                    start_pos = end_pos
                
                # Configure search highlight
                self.text_area.tag_configure("search", background="yellow")
        
        find_button = tk.Button(find_dialog, text="Find All", command=find_text)
        find_button.grid(row=1, column=1, padx=5, pady=5, sticky="e")
        
        # Bind Enter key
        find_entry.bind("<Return>", lambda e: find_text())
    
    def show_language_reference(self):
        """Show language reference dialog"""
        ref_dialog = tk.Toplevel(self.root)
        ref_dialog.title("Minimal Language Reference")
        ref_dialog.geometry("600x500")
        ref_dialog.transient(self.root)
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(ref_dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Overview tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        
        overview_text = scrolledtext.ScrolledText(overview_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        overview_text.pack(fill=tk.BOTH, expand=True)
        overview_text.insert("1.0", """# Minimal Programming Language

Minimal is a clean, simple programming language designed for readability and ease of use.

## Core Philosophy
- Minimize syntax noise
- Make code readable by humans first
- Focus on common use cases
- Provide helpful error messages

## File Extension
.min

## Basic Example:
```
# This is a comment
func main() {
    var message = "Hello, Minimal!"
    print(message)
    
    var n = 10
    if n > 5 {
        print("n is greater than 5")
    }
}
```
""")
        overview_text.config(state=tk.DISABLED)
        
        # Syntax tab
        syntax_frame = ttk.Frame(notebook)
        notebook.add(syntax_frame, text="Syntax")
        
        syntax_text = scrolledtext.ScrolledText(syntax_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        syntax_text.pack(fill=tk.BOTH, expand=True)
        syntax_text.insert("1.0", """## Syntax

### Variables
```
var name = value
```

### Functions
```
func name(param1, param2) {
    # function body
    return value
}
```

### Control Flow
```
if condition {
    # code
} else {
    # code
}

loop i from 0 to 10 {
    # code
}

loop condition {
    # code
}
```

### Data Types
- num: Numbers (integer or float)
- str: Strings
- bool: Boolean values
- list: Lists of values
- map: Key-value pairs
```
var count = 42          # num
var name = "Alice"      # str
var active = true       # bool
var items = [1, 2, 3]   # list
var user = {            # map
    "name": "Alice", 
    "age": 30
}
```
""")
        syntax_text.config(state=tk.DISABLED)
        
        # Keywords tab
        keywords_frame = ttk.Frame(notebook)
        notebook.add(keywords_frame, text="Keywords")
        
        keywords_text = scrolledtext.ScrolledText(keywords_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        keywords_text.pack(fill=tk.BOTH, expand=True)
        keywords_text.insert("1.0", """## Keywords

- func: Define a function
- var: Declare a variable
- if: Conditional statement
- else: Alternative branch in conditional
- loop: Iteration construct
- return: Return a value from a function
- true: Boolean true value
- false: Boolean false value
- null: Null value
- break: Exit a loop
- continue: Skip to next iteration
- and: Logical AND
- or: Logical OR
- not: Logical NOT
- import: Import a module
- export: Export definitions

## Types
- num: Number type
- str: String type
- bool: Boolean type
- list: List type
- map: Map type
""")
        keywords_text.config(state=tk.DISABLED)
        
        # Examples tab
        examples_frame = ttk.Frame(notebook)
        notebook.add(examples_frame, text="Examples")
        
        examples_text = scrolledtext.ScrolledText(examples_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        examples_text.pack(fill=tk.BOTH, expand=True)
        examples_text.insert("1.0", """## Examples

### Hello World
```
func main() {
    print("Hello, world!")
}
```

### Factorial Function
```
func factorial(n) {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

func main() {
    var result = factorial(5)
    print("Factorial of 5 is: " + result)
}
```

### List Operations
```
func main() {
    var numbers = [1, 2, 3, 4, 5]
    
    # Print all numbers
    loop i from 0 to len(numbers) {
        print(numbers[i])
    }
    
    # Double each number
    var doubled = []
    loop num in numbers {
        doubled.append(num * 2)
    }
    
    print("Doubled: " + doubled)
}
```
""")
        examples_text.config(state=tk.DISABLED)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Minimal Language Editor

A clean, minimalist code editor for the Minimal programming language.

Features:
- Syntax highlighting
- Code autocompletion
- Line numbering
- File operations
- Basic find functionality

Created with Python and Tkinter.
"""
        messagebox.showinfo("About", about_text)


def main():
    root = tk.Tk()
    editor = CodeEditor(root)
    
    # Center window on screen
    window_width = 1000
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()