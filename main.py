import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QMainWindow, QPushButton
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt
from editor import MainWindow as EditorMainWindow
from settings import Settings

class TitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(30)
        self.setStyleSheet("background-color: #2d2d30; border: none;")
        
        self.titleLabel = QLabel("Python IDE", self)
        self.titleLabel.setStyleSheet("color: #d4d4d4; font: bold 14px;")
        
        self.minimizeButton = QPushButton("_", self)
        self.minimizeButton.setFixedSize(30, 30)
        self.minimizeButton.setStyleSheet("QPushButton { background-color: #2d2d30; color: #d4d4d4; border: none; }"
                                        "QPushButton:hover { background-color: #6a9955; }")
        self.minimizeButton.clicked.connect(self.parent.showMinimized)
        
        self.maximizeButton = QPushButton("□", self)
        self.maximizeButton.setFixedSize(30, 30)
        self.maximizeButton.setStyleSheet("QPushButton { background-color: #2d2d30; color: #d4d4d4; border: none; }"
                                        "QPushButton:hover { background-color: #4ec9b0; }")
        self.maximizeButton.clicked.connect(self.toggle_maximize)
        
        self.closeButton = QPushButton("X", self)
        self.closeButton.setFixedSize(30, 30)
        self.closeButton.setStyleSheet("QPushButton { background-color: #2d2d30; color: #d4d4d4; border: none; }"
                                    "QPushButton:hover { background-color: #ce9178; }")
        self.closeButton.clicked.connect(self.parent.close)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.addWidget(self.titleLabel)
        layout.addStretch()
        layout.addWidget(self.minimizeButton)
        layout.addWidget(self.maximizeButton)
        layout.addWidget(self.closeButton)
        self.setLayout(layout)
        
        self.isMaximized = False
        self.dragPos = None

    def toggle_maximize(self):
        if self.isMaximized:
            self.parent.showNormal()
            self.isMaximized = False
        else:
            self.parent.showMaximized()
            self.isMaximized = True

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.dragPos:
            self.parent.move(self.parent.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()
            event.accept()

class CustomMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.editorWindow = EditorMainWindow()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.titleBar = TitleBar(self)
        layout.addWidget(self.titleBar)
        layout.addWidget(self.editorWindow)
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(1280, 720)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dynamic palette based on theme
    def apply_theme(theme):
        palette = QPalette()
        if theme == "dark":
            palette.setColor(QPalette.Window, QColor("#1e1e1e"))
            palette.setColor(QPalette.WindowText, QColor("#d4d4d4"))
            palette.setColor(QPalette.Base, QColor("#252526"))
            palette.setColor(QPalette.AlternateBase, QColor("#1e1e1e"))
            palette.setColor(QPalette.ToolTipBase, QColor("#d4d4d4"))
            palette.setColor(QPalette.ToolTipText, QColor("#d4d4d4"))
            palette.setColor(QPalette.Text, QColor("#d4d4d4"))
            palette.setColor(QPalette.Button, QColor("#2d2d30"))
            palette.setColor(QPalette.ButtonText, QColor("#d4d4d4"))
            palette.setColor(QPalette.BrightText, QColor("red"))
            palette.setColor(QPalette.Link, QColor("#2a82da"))
            palette.setColor(QPalette.Highlight, QColor("#2a82da"))
            palette.setColor(QPalette.HighlightedText, QColor("#1e1e1e"))
        else:  # light
            palette.setColor(QPalette.Window, QColor("#ffffff"))
            palette.setColor(QPalette.WindowText, QColor("#000000"))
            palette.setColor(QPalette.Base, QColor("#f5f5f5"))
            palette.setColor(QPalette.AlternateBase, QColor("#ffffff"))
            palette.setColor(QPalette.ToolTipBase, QColor("#000000"))
            palette.setColor(QPalette.ToolTipText, QColor("#000000"))
            palette.setColor(QPalette.Text, QColor("#000000"))
            palette.setColor(QPalette.Button, QColor("#e0e0e0"))
            palette.setColor(QPalette.ButtonText, QColor("#000000"))
            palette.setColor(QPalette.BrightText, QColor("red"))
            palette.setColor(QPalette.Link, QColor("#0000ff"))
            palette.setColor(QPalette.Highlight, QColor("#0078d7"))
            palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        app.setPalette(palette)

    # Apply initial theme
    settings = Settings()
    apply_theme(settings.theme)

    app.setStyleSheet("""
        QMainWindow { border: none; }
        QMenuBar { border: none; }
        QMenuBar::item { padding: 5px; }
        QMenuBar::item:selected { background-color: #0078d7; }
        QStatusBar { border: none; }
        QDockWidget { border: 1px solid #2d2d30; }
        QDockWidget::title { text-align: center; border: none; }
        QTabWidget::pane { border: 1px solid #2d2d30; }
        QTabBar::tab { padding: 5px; border: 1px solid #2d2d30; }
        QTabBar::tab:selected { border: 1px solid #2d2d30; }
        QTreeView { border: none; }
        QPlainTextEdit { border: none; }
        QFontDialog QPushButton {
            border: none;
            font-size: 20px;
            width: 100%;
        }
        QFontDialog QLineEdit {
            border: 1px solid #444;
            padding: 5px;
        }
        QFontDialog QPushButton:hover {
            background-color: #4ec9b0;
        }
        QFontDialog QLabel { }
        QFontDialog QGroupBox { border-color: #2d2d30; }
        QFontDialog QComboBox {
            border: none;
            font-size: 12px;
        }
        QComboBox {
            border: 1px solid #2d2d30;
            padding: 5px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
        }
        QComboBox QAbstractItemView {
            selection-background-color: #0078d7;
        }
        QScrollBar:vertical {
            width: 12px;
            margin: 0px;
            border: 1px solid #2d2d30;
        }
        QScrollBar::handle:vertical {
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            background: none;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        QScrollBar:horizontal {
            height: 12px;
            margin: 0px;
            border: 1px solid #2d2d30;
        }
        QScrollBar::handle:horizontal {
            min-width: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            background: none;
        }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: none;
        }
    """)

    window = CustomMainWindow()
    window.show()
    sys.exit(app.exec_())