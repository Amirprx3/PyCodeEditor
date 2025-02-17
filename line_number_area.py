from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QRect, QSize
from PyQt5.QtGui import QPainter, QFont


class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.codeEditor = editor
        self.setFont(QFont("Consolas", 12))

    def sizeHint(self):
        return QSize(self.codeEditor.lineNumberAreaWidth(), 0)

    def paintEvent(self, event):
        self.codeEditor.lineNumberAreaPaintEvent(event)
