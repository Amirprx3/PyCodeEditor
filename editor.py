#MIT License

#Copyright (c) 2025 Amirprx3

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import os
import sys
import traceback
import jedi
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QTabWidget
from PyQt5.QtWidgets import QSplitter
from PyQt5.QtWidgets import QDockWidget
from PyQt5.QtWidgets import QTreeView
from PyQt5.QtWidgets import QFileSystemModel
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QCompleter
from PyQt5.QtWidgets import QFontDialog

from PyQt5.QtGui import QFont, QColor, QIcon, QPainter
from PyQt5.QtCore import Qt, QDir, QModelIndex, QRect, QStringListModel

from terminal import Terminal
from syntax_highlighter import PythonSyntaxHighlighter
from line_number_area import LineNumberArea
from settings import Settings


EDITOR_BACKGROUND    = "#1e1e1e"
EDITOR_FOREGROUND    = "#d4d4d4"
LINE_NUMBER_BG       = "#2d2d30"
LINE_NUMBER_FG       = "#858585"
FILE_EXPLORER_BG     = "#252526"
FILE_EXPLORER_FG     = "#d4d4d4"
COMPLETER_BG         = "black"
COMPLETER_FG         = "white"


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Fira Code", 12))

        self.setStyleSheet(f"background-color: {EDITOR_BACKGROUND}; color: {EDITOR_FOREGROUND};")
        self.highlighter = PythonSyntaxHighlighter(self.document())
        self._setup_line_numbers()

        self.completer = QCompleter()
        self.completer.setWidget(self)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.activated.connect(self.insert_completion)
        self.completer.setModel(QStringListModel())
        self.completer.popup().setStyleSheet(f"background-color: {COMPLETER_BG}; color: {COMPLETER_FG};")

    def _setup_line_numbers(self):
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)

    def lineNumberAreaWidth(self):
        digits = len(str(max(1, self.blockCount())))
        space = 3 + self.fontMetrics().width('9') * digits
        return space

    def updateLineNumberAreaWidth(self, _):
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def updateLineNumberArea(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.updateLineNumberAreaWidth(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height())
        )

    def lineNumberAreaPaintEvent(self, event):
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(LINE_NUMBER_BG))
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.setPen(QColor(LINE_NUMBER_FG))
                painter.drawText(
                    0,
                    int(top),
                    self.line_number_area.width(),
                    self.fontMetrics().height(),
                    Qt.AlignRight,
                    number
                )
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            blockNumber += 1

    def insert_completion(self, completion):
        cursor = self.textCursor()
        extra = len(completion) - len(self.completer.completionPrefix())
        if extra > 0:
            cursor.insertText(completion[-extra:])
        self.setTextCursor(cursor)

    def textUnderCursor(self):
        cursor = self.textCursor()
        cursor.select(cursor.WordUnderCursor)
        return cursor.selectedText()

    def keyPressEvent(self, event):
        if self.completer.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape, Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return
        super().keyPressEvent(event)
        self.trigger_completion(event)

    def trigger_completion(self, event):
        if event.text() and (event.text().isalnum() or event.text() in ('_', '.')):
            self.show_completion_suggestions()

    def show_completion_suggestions(self):
        cursor = self.textCursor()
        pos = cursor.position()
        block = cursor.block()
        line = block.blockNumber() + 1
        column = pos - block.position()
        source = self.toPlainText()
        try:
            script = jedi.Script(source, path='')
            completions = script.complete(line, column)
            suggestions = [comp.name for comp in completions]
            if suggestions:
                model = QStringListModel()
                model.setStringList(suggestions)
                self.completer.setModel(model)
                prefix = self.textUnderCursor()
                self.completer.setCompletionPrefix(prefix)
                cr = self.cursorRect()
                cr.setWidth(
                    self.completer.popup().sizeHintForColumn(0) +
                    self.completer.popup().verticalScrollBar().sizeHint().width()
                )
                self.completer.complete(cr)
            else:
                self.completer.popup().hide()
        except Exception as e:
            print("Completion error:", e)
            self.completer.popup().hide()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python IDE")
        self.setGeometry(100, 100, 1280, 720)
        self._setup_ui()
        self._setup_actions()
        self._setup_file_explorer()
        self._setup_status_bar()
        self.settings = Settings()

    def _setup_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)

        # Editor and Terminal Splitter
        editor_terminal_splitter = QSplitter(Qt.Vertical)

        # Initialize editor_tabs
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setStyleSheet(f"background-color: {EDITOR_BACKGROUND}; color: {EDITOR_FOREGROUND}; border: 1px solid #2d2d30;")
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self.close_tab)

        # Apply settings to editors after initializing editor_tabs
        for i in range(self.editor_tabs.count()):
            editor = self.editor_tabs.widget(i)
            if isinstance(editor, CodeEditor):
                editor.setFont(self.settings.editor_font)

        self.terminal = Terminal()
        self.terminal.setStyleSheet(f"background-color: {EDITOR_BACKGROUND}; color: {EDITOR_FOREGROUND}; border: 1px solid #2d2d30;")
        editor_terminal_splitter.addWidget(self.editor_tabs)
        editor_terminal_splitter.addWidget(self.terminal)
        main_splitter.addWidget(editor_terminal_splitter)
        self.setCentralWidget(main_splitter)
        
    def _setup_actions(self):
        file_menu = self.menuBar().addMenu("&File")
        new_action = QAction(QIcon.fromTheme("document-new"), "New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)

        open_file_action = QAction(QIcon.fromTheme("document-open"), "Open File...", self)
        open_file_action.setShortcut("Ctrl+O")
        open_file_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_file_action)

        open_folder_action = QAction("Open Folder...", self)
        open_folder_action.setShortcut("Ctrl+K Ctrl+O")
        open_folder_action.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(open_folder_action)

        save_action = QAction(QIcon.fromTheme("document-save"), "Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        run_menu = self.menuBar().addMenu("&Run")
        run_action = QAction("Run Code", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self.run_code)
        run_menu.addAction(run_action)

        terminal_menu = self.menuBar().addMenu("&Terminal")
        open_terminal_action = QAction("Open System Terminal", self)
        open_terminal_action.triggered.connect(self.open_system_terminal)
        terminal_menu.addAction(open_terminal_action)

        settings_menu = self.menuBar().addMenu(" &Settings")
        # Font settings action
        font_settings_action = QAction("Font Settings", self)
        font_settings_action.triggered.connect(self.show_font_settings)
        settings_menu.addAction(font_settings_action)

    def show_font_settings(self):
        # Check if there is an active editor tab
        current_editor = self.editor_tabs.currentWidget()
        if current_editor is None:
            # Use default font if no tab is open
            default_font = QFont("Fira Code", 12)
            font, ok = QFontDialog.getFont(default_font, self, "Choose Font")
        else:
            # Use the font of the current editor if a tab is open
            font, ok = QFontDialog.getFont(current_editor.font(), self, "Choose Font")

        if ok:
            # Apply the selected font to all open tabs
            for i in range(self.editor_tabs.count()):
                editor = self.editor_tabs.widget(i)
                if isinstance(editor, CodeEditor):  # Ensure the widget is a CodeEditor
                    editor.setFont(font)

            # Save the selected font to settings
            self.settings.editor_font = font
            self.settings.save_settings()

    def _setup_file_explorer(self):
        self.file_explorer = QDockWidget("File Explorer", self)
        self.file_model = QFileSystemModel()
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.doubleClicked.connect(self.open_file)
        self.tree_view.setStyleSheet(f"background-color: {FILE_EXPLORER_BG}; color: {FILE_EXPLORER_FG};")
        self.tree_view.setStyleSheet("""
            QHeaderView::section {
                background-color: #252526;  
                color: #d4d4d4;  
                border: 1px solid #2d2d30; 
            }
        """)
        container = QVBoxLayout()
        container.addWidget(self.tree_view)
        container_widget = QWidget()
        container_widget.setLayout(container)
        self.file_explorer.setWidget(container_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_explorer)

        # Initially set the root index to an invalid index (empty)
        self.tree_view.setRootIndex(QModelIndex())

    def _setup_status_bar(self):
        self.statusBar().showMessage("Ready")

    def new_file(self):
        editor = CodeEditor(self)
        editor.setFont(self.settings.editor_font)
        self.editor_tabs.addTab(editor, "Untitled.py")
        self.editor_tabs.setCurrentWidget(editor)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            self.open_file(file_path)

    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", QDir.currentPath())
        if folder_path:
            try:
                if os.path.isdir(folder_path) and os.access(folder_path, os.R_OK):
                    self.file_model.setRootPath(folder_path)
                    self.tree_view.setRootIndex(self.file_model.index(folder_path))
                    self.statusBar().showMessage(f"Opened folder: {folder_path}")
                else:
                    QMessageBox.warning(self, "Error", f"The selected folder is not accessible: {folder_path}")
            except Exception as e:
                print(traceback.format_exc())
                QMessageBox.critical(self, "Critical Error", f"Failed to open folder: {e}")

    def open_file(self, path):
        if isinstance(path, QModelIndex):
            path = self.file_model.filePath(path)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            editor = CodeEditor(self)
            editor.setFont(self.settings.editor_font)
            editor.setPlainText(content)
            self.editor_tabs.addTab(editor, os.path.basename(path))
            self.editor_tabs.setCurrentWidget(editor)
            self.statusBar().showMessage(f"Opened file: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"Failed to open file: {e}")

    def save_file(self):
        current_editor = self.editor_tabs.currentWidget()
        if current_editor:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save File")
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(current_editor.toPlainText())
                    self.editor_tabs.setTabText(self.editor_tabs.currentIndex(), os.path.basename(file_path))
                    self.statusBar().showMessage(f"Saved file: {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    def close_tab(self, index):
        self.editor_tabs.removeTab(index)

    def run_code(self):
        current_editor = self.editor_tabs.currentWidget()
        if current_editor:
            try:
                code = current_editor.toPlainText()
                with open("temp_code.py", "w", encoding='utf-8') as file:
                    file.write(code)
                self.terminal.run_command("python temp_code.py")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run code: {e}")

    def open_system_terminal(self):
        os.system("start cmd")  
