from PyQt5.QtGui import QFont
from PyQt5.QtCore import QSettings

class Settings:
    def __init__(self):
        self.settings = QSettings("PythonIDE", "Settings")
    
        self.load_settings()

    def load_settings(self):
        # Load font settings
        font_family = self.settings.value("font/family", "Fira Code")
        font_size = int(self.settings.value("font/size", 12))
        self.editor_font = QFont(font_family, font_size)
        
        
    def save_settings(self):
        # Save font settings
        self.settings.setValue("font/family", self.editor_font.family())
        self.settings.setValue("font/size", self.editor_font.pointSize())
