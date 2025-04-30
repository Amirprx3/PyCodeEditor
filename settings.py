from PyQt5.QtGui import QFont
from PyQt5.QtCore import QSettings

# Theme colors
THEMES = {
    "dark": {
        "EDITOR_BACKGROUND": "#1e1e1e",
        "EDITOR_FOREGROUND": "#d4d4d4",
        "LINE_NUMBER_BG": "#2d2d30",
        "LINE_NUMBER_FG": "#858585",
        "FILE_EXPLORER_BG": "#252526",
        "FILE_EXPLORER_FG": "#d4d4d4",
        "COMPLETER_BG": "black",
        "COMPLETER_FG": "white",
        "AI_PANEL_BG": "#252526",
        "AI_PANEL_FG": "#d4d4d4",
        "TERMINAL_BG": "#1e1e1e",
        "TERMINAL_FG": "#d4d4d4"
    },
    "light": {
        "EDITOR_BACKGROUND": "#ffffff",
        "EDITOR_FOREGROUND": "#000000",
        "LINE_NUMBER_BG": "#f0f0f0",
        "LINE_NUMBER_FG": "#666666",
        "FILE_EXPLORER_BG": "#f5f5f5",
        "FILE_EXPLORER_FG": "#000000",
        "COMPLETER_BG": "#ffffff",
        "COMPLETER_FG": "#000000",
        "AI_PANEL_BG": "#f5f5f5",
        "AI_PANEL_FG": "#000000",
        "TERMINAL_BG": "#ffffff",
        "TERMINAL_FG": "#000000"
    }
}

class Settings:
    def __init__(self):
        self.settings = QSettings("PythonIDE", "Settings")
        self.load_settings()

    def load_settings(self):
        # Load font settings
        font_family = self.settings.value("font/family", "Fira Code")
        font_size = int(self.settings.value("font/size", 12))
        self.editor_font = QFont(font_family, font_size)
        
        # Load Python interpreter path
        self.python_interpreter = self.settings.value("python/interpreter", "python")
        
        # Load theme
        self.theme = self.settings.value("theme", "dark")

    def save_settings(self):
        # Save font settings
        self.settings.setValue("font/family", self.editor_font.family())
        self.settings.setValue("font/size", self.editor_font.pointSize())
        
        # Save Python interpreter path
        self.settings.setValue("python/interpreter", self.python_interpreter)
        
        # Save theme
        self.settings.setValue("theme", self.theme)