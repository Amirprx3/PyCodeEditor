import os
import traceback
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPlainTextEdit, QAction,
    QFileDialog, QTabWidget, QSplitter, QDockWidget, QTreeView,
    QFileSystemModel, QLabel, QPushButton, QHBoxLayout, QStatusBar, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QFont, QColor, QIcon, QPalette, QPainter
from PyQt5.QtCore import Qt, QDir, QTimer, QSettings, QModelIndex, QRect

from terminal import Terminal
from syntax_highlighter import PythonSyntaxHighlighter
from line_number_area import LineNumberArea


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Fira Code", 12))
        self.highlighter = PythonSyntaxHighlighter(self.document())
        self._setup_line_numbers()

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
        painter.fillRect(event.rect(), Qt.lightGray)

        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.setPen(Qt.black)
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python IDE")
        self.setGeometry(100, 100, 1280, 720)
        self._setup_ui()
        self._setup_actions()
        self._setup_file_explorer()
        self._setup_status_bar()

    def _setup_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)

        # File Explorer with toolbar
        self.file_explorer = QDockWidget("File Explorer", self)
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(QDir.rootPath())  # Use root path to ensure accessibility

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.doubleClicked.connect(self.open_file)

        container = QVBoxLayout()
        container.addWidget(self.tree_view)

        container_widget = QWidget()
        container_widget.setLayout(container)
        self.file_explorer.setWidget(container_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_explorer)

        # Editor and Terminal
        editor_terminal_splitter = QSplitter(Qt.Vertical)
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self.close_tab)
        self.terminal = Terminal()
        editor_terminal_splitter.addWidget(self.editor_tabs)
        editor_terminal_splitter.addWidget(self.terminal)

        main_splitter.addWidget(editor_terminal_splitter)
        self.setCentralWidget(main_splitter)

    def _setup_actions(self):
        file_menu = self.menuBar().addMenu("&File")

        # New File Action
        new_action = QAction(QIcon.fromTheme("document-new"), "New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)

        # Open File Action
        open_file_action = QAction(QIcon.fromTheme("document-open"), "Open File...", self)
        open_file_action.setShortcut("Ctrl+O")
        open_file_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_file_action)

        # Open Folder Action
        open_folder_action = QAction("Open Folder...", self)
        open_folder_action.setShortcut("Ctrl+K Ctrl+O")
        open_folder_action.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(open_folder_action)

        # Save File Action
        save_action = QAction(QIcon.fromTheme("document-save"), "Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        # Run Menu
        run_menu = self.menuBar().addMenu("&Run")
        run_action = QAction("Run Code", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self.run_code)
        run_menu.addAction(run_action)

        # Terminal Menu
        terminal_menu = self.menuBar().addMenu("&Terminal")
        open_terminal_action = QAction("Open System Terminal", self)
        open_terminal_action.triggered.connect(self.open_system_terminal)
        terminal_menu.addAction(open_terminal_action)

    def _setup_file_explorer(self):
        self.tree_view.setRootIndex(self.file_model.index(QDir.currentPath()))
        self.tree_view.setColumnWidth(0, 250)

    def _setup_status_bar(self):
        self.statusBar().showMessage("Ready")

    def new_file(self):
        editor = CodeEditor(self)
        self.editor_tabs.addTab(editor, "Untitled.py")
        self.editor_tabs.setCurrentWidget(editor)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Python Files (*.py);;All Files (*)")
        if file_path:
            self.open_file(file_path)

    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", QDir.currentPath())
        if folder_path:
            try:
                print(f"Selected folder: {folder_path}")  # Add logging
                if os.path.isdir(folder_path) and os.access(folder_path, os.R_OK):
                    self.file_model.setRootPath(folder_path)
                    self.tree_view.setRootIndex(self.file_model.index(folder_path))
                    self.statusBar().showMessage(f"Opened folder: {folder_path}")
                else:
                    QMessageBox.warning(self, "Error", f"The selected folder is not accessible: {folder_path}")
            except Exception as e:
                error_message = traceback.format_exc()
                print(f"Error opening folder: {error_message}")  # Add logging
                QMessageBox.critical(self, "Critical Error", f"Failed to open folder: {e}")
    def open_file(self, path):
        if isinstance(path, QModelIndex):
            path = self.file_model.filePath(path)

        if os.path.isfile(path) and os.access(path, os.R_OK):  # Check if the file is readable
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()

                editor = CodeEditor(self)
                editor.setPlainText(content)
                self.editor_tabs.addTab(editor, os.path.basename(path))
                self.editor_tabs.setCurrentWidget(editor)
                self.statusBar().showMessage(f"Opened file: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Critical Error", f"Failed to open file: {e}")
        else:
            QMessageBox.warning(self, "Warning", f"The selected file is not accessible: {path}")

    def save_file(self):
        current_editor = self.editor_tabs.currentWidget()
        if current_editor:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save File")
            if file_path:
                with open(file_path, 'w') as file:
                    file.write(current_editor.toPlainText())
                self.editor_tabs.setTabText(self.editor_tabs.currentIndex(), os.path.basename(file_path))
                self.statusBar().showMessage(f"Saved file: {file_path}")

    def close_tab(self, index):
        self.editor_tabs.removeTab(index)

    def run_code(self):
        current_editor = self.editor_tabs.currentWidget()
        if current_editor:
            code = current_editor.toPlainText()
            with open("temp_code.py", "w") as file:
                file.write(code)
            self.terminal.run_command("python temp_code.py")

    def open_system_terminal(self):
        import os
        os.system("start cmd")  # Open Windows Command Prompt

    def create_file(self):
        index = self.tree_view.currentIndex()
        if not index.isValid():
            return

        path = self.file_model.filePath(index)
        if os.path.isdir(path):
            file_name, ok = QInputDialog.getText(self, "New File", "Enter file name:")
            if ok and file_name:
                file_path = os.path.join(path, file_name)
                try:
                    open(file_path, 'w').close()  # Create an empty file
                    self.file_model.refresh()
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to create file: {e}")

    def create_folder(self):
        index = self.tree_view.currentIndex()
        if not index.isValid():
            return

        path = self.file_model.filePath(index)
        if os.path.isdir(path):
            folder_name, ok = QInputDialog.getText(self, "New Folder", "Enter folder name:")
            if ok and folder_name:
                folder_path = os.path.join(path, folder_name)
                try:
                    os.mkdir(folder_path)
                    self.file_model.refresh()
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to create folder: {e}")

    def show_current_path(self):
        index = self.tree_view.currentIndex()
        if index.isValid():
            path = self.file_model.filePath(index)
            QMessageBox.information(self, "Current Path", f"Selected Path: {path}")

    def open_file(self, index):
        path = self.file_model.filePath(index)
        if os.path.isfile(path):
            with open(path, 'r') as file:
                content = file.read()
            editor = CodeEditor(self)
            editor.setPlainText(content)
            self.editor_tabs.addTab(editor, os.path.basename(path))
            self.editor_tabs.setCurrentWidget(editor)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Python Files (*.py);;All Files (*)")
        if file_path:
            self.open_file(file_path)

    def open_file(self, path):
        if isinstance(path, QModelIndex):
            path = self.file_model.filePath(path)
        with open(path, 'r') as file:
            content = file.read()
        editor = CodeEditor(self)
        editor.setPlainText(content)
        self.editor_tabs.addTab(editor, os.path.basename(path))
        self.editor_tabs.setCurrentWidget(editor)

    def save_file(self):
        current_editor = self.editor_tabs.currentWidget()
        if current_editor:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save File")
            if file_path:
                with open(file_path, 'w') as file:
                    file.write(current_editor.toPlainText())
                self.editor_tabs.setTabText(self.editor_tabs.currentIndex(), os.path.basename(file_path))

    def close_tab(self, index):
        self.editor_tabs.removeTab(index)

    def run_code(self):
        current_editor = self.editor_tabs.currentWidget()
        if current_editor:
            code = current_editor.toPlainText()
            with open("temp_code.py", "w") as file:
                file.write(code)
            self.terminal.run_command("python temp_code.py")