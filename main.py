
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QMainWindow, QPushButton
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt
from editor import MainWindow as EditorMainWindow


class TitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(30)
        self.setStyleSheet("background-color: #2d2d30; border: none;")
        
        # title label
        self.titleLabel = QLabel("Python IDE", self)
        self.titleLabel.setStyleSheet("color: #d4d4d4; font: bold 14px;")
        
        # add buttons
        self.minimizeButton = QPushButton("_", self)  # minimize butten
        self.minimizeButton.setFixedSize(30, 30)
        self.minimizeButton.setStyleSheet("QPushButton { background-color: #2d2d30; color: #d4d4d4; border: none; }"
                                        "QPushButton:hover { background-color: #6a9955; }")
        self.minimizeButton.clicked.connect(self.parent.showMinimized)
        
        self.maximizeButton = QPushButton("□", self)  # maximize butten
        self.maximizeButton.setFixedSize(30, 30)
        self.maximizeButton.setStyleSheet("QPushButton { background-color: #2d2d30; color: #d4d4d4; border: none; }"
                                        "QPushButton:hover { background-color: #4ec9b0; }")
        self.maximizeButton.clicked.connect(self.toggle_maximize)
        
        self.closeButton = QPushButton("X", self)  # close button
        self.closeButton.setFixedSize(30, 30)
        self.closeButton.setStyleSheet("QPushButton { background-color: #2d2d30; color: #d4d4d4; border: none; }"
                                    "QPushButton:hover { background-color: #ce9178; }")
        self.closeButton.clicked.connect(self.parent.close)
        
        # layout settigs
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.addWidget(self.titleLabel)
        layout.addStretch()
        layout.addWidget(self.minimizeButton)
        layout.addWidget(self.maximizeButton)
        layout.addWidget(self.closeButton)
        self.setLayout(layout)
        
        # Variable to manage zoom mode
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


    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor("#1e1e1e"))
    dark_palette.setColor(QPalette.WindowText, QColor("#d4d4d4"))
    dark_palette.setColor(QPalette.Base, QColor("#252526"))
    dark_palette.setColor(QPalette.AlternateBase, QColor("#1e1e1e"))
    dark_palette.setColor(QPalette.ToolTipBase, QColor("#d4d4d4"))
    dark_palette.setColor(QPalette.ToolTipText, QColor("#d4d4d4"))
    dark_palette.setColor(QPalette.Text, QColor("#d4d4d4"))
    dark_palette.setColor(QPalette.Button, QColor("#2d2d30"))
    dark_palette.setColor(QPalette.ButtonText, QColor("#d4d4d4"))
    dark_palette.setColor(QPalette.BrightText, QColor("red"))
    dark_palette.setColor(QPalette.Link, QColor("#2a82da"))
    dark_palette.setColor(QPalette.Highlight, QColor("#2a82da"))
    dark_palette.setColor(QPalette.HighlightedText, QColor("#1e1e1e"))
    
    app.setPalette(dark_palette)


    app.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; border: none; }
        QMenuBar { background-color: #2d2d30; color: #d4d4d4; border: none; }
        QMenuBar::item { background-color: #2d2d30; color: #d4d4d4; }
        QMenuBar::item:selected { background-color: #2a82da; }
        QStatusBar { background-color: #2d2d30; color: #d4d4d4; border: none; }
        QDockWidget { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #2d2d30; }
        QDockWidget::title { background-color: #2d2d30; text-align: center; border: none; }
        QTabWidget::pane { background-color: #1e1e1e; border: 1px solid #2d2d30; }
        QTabBar::tab { background-color: #2d2d30; color: #d4d4d4; padding: 5px; border: 1px solid #2d2d30; }
        QTabBar::tab:selected { background-color: #1e1e1e; border: 1px solid #2d2d30; }
        QTreeView { background-color: #252526; color: #d4d4d4; border: none; }
        QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: none; }
        QFontDialog QPushButton {
            background-color: #2d2d30;
            color: #d4d4d4;
            border: none;
            font-size: 20px;
            width : 100%;
        }
                      
        QFontDialog QLineEdit {
            background-color: #2d2d30;
            color: #d4d4d4;
            border: 1px solid #444;
            padding: 5px;
        }
        
        QFontDialog QPushButton:hover {
            background-color: #4ec9b0;
        }
                      
        QFontDialog QLabel {
            color: #d4d4d4; 
        }

        QFontDialog QGroupBox {
            background-color: #2d2d30; 
            color: #d4d4d4; 
            border-color : #2d2d30;
        }
        
                      
        QFontDialog QComboBox{
            background-color: #2d2d30;
            color: #d4d4d4;
            border: none;
            font-size: 12px;
        }
        
        /* Scroll bar styles for vertical and horizontal scroll bars */
        QScrollBar:vertical {
            background: #1e1e1e;
            width: 12px;
            margin: 0px;
            border: 1px solid #2d2d30;
        }
        QScrollBar::handle:vertical {
            background: #2d2d30;
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
            background: #1e1e1e;
            height: 12px;
            margin: 0px;
            border: 1px solid #2d2d30;
        }
        QScrollBar::handle:horizontal {
            background: #2d2d30;
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
