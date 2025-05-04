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
import re
import jedi
import requests
import time
import html
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters.html import HtmlFormatter
import subprocess
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPlainTextEdit, QAction, QTabWidget, QSplitter, QDockWidget, QTreeView, QFileSystemModel, QFileDialog, QMessageBox, QCompleter, QFontDialog, QMenu, QTextEdit, QLineEdit, QComboBox, QToolBar, QInputDialog
from PyQt5.QtGui import QFont, QColor, QIcon, QPainter, QStandardItemModel, QStandardItem, QKeySequence
from PyQt5.QtCore import Qt, QDir, QModelIndex, QRect, QStringListModel
from PyQt5.QtCore import QSettings
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters.html import HtmlFormatter
from pygments.styles.monokai import MonokaiStyle
from pygments.token import Comment,Name,Keyword, Token
from terminal import Terminal
from syntax_highlighter import PythonSyntaxHighlighter
from line_number_area import LineNumberArea
from settings import Settings, THEMES




class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None, theme="dark"):
        super().__init__(parent)
        self.theme = theme
        self.setFont(QFont("Fira Code", 12))
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(' ') * 4)

        self.apply_theme()
        self.highlighter = PythonSyntaxHighlighter(self.document())
        self._setup_line_numbers()
            
        try:
            self.completer = QCompleter()
            self.completer.setWidget(self)
            self.completer.setCompletionMode(QCompleter.PopupCompletion)
            self.completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.completer.activated.connect(self.insert_completion)
            self.completer.setModel(QStandardItemModel())
            self.completer.popup().setStyleSheet(f"background-color: {THEMES[self.theme]['COMPLETER_BG']}; color: {THEMES[self.theme]['COMPLETER_FG']};")
            print("Completer initialized successfully")
        except Exception as e:
            print(f"Error initializing completer: {e}")
            raise
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def _setup_line_numbers(self):
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)

    def apply_theme(self):
        self.setStyleSheet(f"background-color: {THEMES[self.theme]['EDITOR_BACKGROUND']}; color: {THEMES[self.theme]['EDITOR_FOREGROUND']};")
        if hasattr(self, 'completer'):
            self.completer.popup().setStyleSheet(f"background-color: {THEMES[self.theme]['COMPLETER_BG']}; color: {THEMES[self.theme]['COMPLETER_FG']};")

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
        painter.fillRect(event.rect(), QColor(THEMES[self.theme]['LINE_NUMBER_BG']))
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.setPen(QColor(THEMES[self.theme]['LINE_NUMBER_FG']))
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
        if hasattr(self, 'completer') and self.completer.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape, Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Space:
            print("Ctrl+Space pressed, triggering completion")
            self.show_completion_suggestions()
            event.accept()
            return
        super().keyPressEvent(event)
        self.trigger_completion(event)

    def trigger_completion(self, event):
        if event.text() and (event.text().isalnum() or event.text() in ('_', '.')):
            print("Triggering completion for text:", event.text())
            self.show_completion_suggestions()

    def show_completion_suggestions(self):
        if not hasattr(self, 'completer'):
            print("Completer not defined, skipping completion")
            return
        
        cursor = self.textCursor()
        pos = cursor.position()
        block = cursor.block()
        line = block.blockNumber() + 1
        column = pos - block.position()
        source = self.toPlainText()
        try:
            block_text = block.text().strip()
            if block_text.startswith('import ') or block_text.startswith('from '):
                if ' as ' in block_text:
                    cursor_word = self.textUnderCursor()
                    if cursor_word in ('as',):
                        print("Skipping completion after 'as' in import")
                        self.completer.popup().hide()
                        return
            
            script = jedi.Script(source, path=self.parentWidget().Current_FileName if hasattr(self.parentWidget(), 'Current_FileName') else '')
            completions = script.complete(line, column, fuzzy=True)
            model = QStandardItemModel()
            for comp in completions:
                item = QStandardItem(comp.name)
                item.setData(f"{comp.type}: {comp.description}", Qt.ToolTipRole)
                model.appendRow(item)
            self.completer.setModel(model)
            prefix = self.textUnderCursor()
            self.completer.setCompletionPrefix(prefix)
            cr = self.cursorRect()
            cr.setWidth(
                self.completer.popup().sizeHintForColumn(0) +
                self.completer.popup().verticalScrollBar().sizeHint().width()
            )
            print(f"Showing completions for prefix: {prefix}")
            self.completer.complete(cr)
        except Exception as e:
            print("Completion error:", e)
            self.completer.popup().hide()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
        QMenu {
            background-color: #2d2d30;
            color: #d4d4d4;
            border: 1px solid #444;
        }
        QMenu::item {
            padding: 5px 20px;
            background-color: transparent;
        }
        QMenu::item:selected {
            background-color: #4ec9b0;  /* your hover color */
            color: black;
        }
        """)
        if self.textCursor().hasSelection():
            ai_action = QAction("Ask AI", self)
            ai_action.triggered.connect(self.open_ai_panel)
            menu.addAction(ai_action)
        menu.exec_(self.mapToGlobal(pos))

    def open_ai_panel(self):
        main_window = self.parentWidget()
        while main_window and not isinstance(main_window, MainWindow):
            main_window = main_window.parentWidget()
        if main_window:
            main_window.show_ai_panel(self.textCursor().selectedText())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python IDE")
        self.setGeometry(100, 100, 1280, 720)
        self.settings = Settings()
        self._setup_ui()
        self._setup_actions()
        self._setup_file_explorer()
        self._setup_status_bar()
        self.ai_dock = None
        self.terminal_visible = True
    
    def reset_to_defaults(self):
        self.settings.settings.clear()  # Clear internal QSettings
        self.settings.load_settings()  # Reload default settings

        # Apply theme and font updates
        self.apply_theme_to_file_explorer()
        self.apply_theme_to_tabs()
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.settings.settings.clear()
            self.settings.load_settings()
            for i in range(self.editor_tabs.count()):
                editor = self.editor_tabs.widget(i)
                if isinstance(editor, CodeEditor):
                    editor.setFont(self.settings.editor_font)
                    editor.theme = self.settings.theme
                    editor.apply_theme()

            self.terminal.theme = self.settings.theme
            self.terminal.apply_theme()

            self.statusBar().showMessage("Settings reset to defaults")
    
    def _setup_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        self.file_explorer = QDockWidget("File Explorer", self)
        self.file_model = QFileSystemModel()
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.doubleClicked.connect(self.open_file)
        self.apply_theme_to_file_explorer()
        container = QVBoxLayout()
        container.addWidget(self.tree_view)
        container_widget = QWidget()
        container_widget.setLayout(container)
        self.file_explorer.setWidget(container_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_explorer)

        editor_terminal_splitter = QSplitter(Qt.Vertical)
        self.editor_tabs = QTabWidget()
        self.apply_theme_to_tabs()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self.close_tab)

        for i in range(self.editor_tabs.count()):
            editor = self.editor_tabs.widget(i)
            if isinstance(editor, CodeEditor):
                editor.setFont(self.settings.editor_font)
                editor.theme = self.settings.theme
                editor.apply_theme()

        self.terminal = Terminal(theme=self.settings.theme)
        editor_terminal_splitter.addWidget(self.editor_tabs)
        editor_terminal_splitter.addWidget(self.terminal)
        main_splitter.addWidget(editor_terminal_splitter)
        self.setCentralWidget(main_splitter)

        self.toolbar = QToolBar("Interpreter")
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.interpreter_combo = QComboBox()
        self.interpreter_combo.setStyleSheet(f"background-color: {THEMES[self.settings.theme]['AI_PANEL_BG']}; color: {THEMES[self.settings.theme]['AI_PANEL_FG']}; border: 1px solid #2d2d30; padding: 5px;")
        self.interpreter_combo.addItem(self.settings.python_interpreter)
        self.interpreter_combo.currentTextChanged.connect(self.change_interpreter)
        self.toolbar.addWidget(self.interpreter_combo)
        self._populate_interpreters()

    def keyPressEvent(self, event):
        if event == QKeySequence("Ctrl+`"):
            self.toggle_terminal()
            event.accept()
        else:
            super().keyPressEvent(event)

    def toggle_terminal(self):
        if self.terminal_visible:
            self.terminal.hide()
            self.terminal_visible = False
            self.statusBar().showMessage("Terminal hidden")
        else:
            self.terminal.show()
            self.terminal_visible = True
            self.statusBar().showMessage("Terminal shown")

    def apply_theme_to_file_explorer(self):
        self.tree_view.setStyleSheet(f"background-color: {THEMES[self.settings.theme]['FILE_EXPLORER_BG']}; color: {THEMES[self.settings.theme]['FILE_EXPLORER_FG']};")
        self.tree_view.setStyleSheet(f"""
            QTreeView {{ background-color: {THEMES[self.settings.theme]['FILE_EXPLORER_BG']}; color: {THEMES[self.settings.theme]['FILE_EXPLORER_FG']}; }}
            QHeaderView::section {{
                background-color: {THEMES[self.settings.theme]['FILE_EXPLORER_BG']};  
                color: {THEMES[self.settings.theme]['FILE_EXPLORER_FG']};  
                border: 1px solid #2d2d30;
            }}
        """)

    def apply_theme_to_tabs(self):
        self.editor_tabs.setStyleSheet(f"background-color: {THEMES[self.settings.theme]['EDITOR_BACKGROUND']}; color: {THEMES[self.settings.theme]['EDITOR_FOREGROUND']}; border: 1px solid #2d2d30;")

    def _populate_interpreters(self):
        interpreters = ["python"]
        try:
            result = subprocess.run(["where", "python"], capture_output=True, text=True)
            for path in result.stdout.splitlines():
                if "python" in path.lower():
                    interpreters.append(path.strip())
            result = subprocess.run(["where", "python3"], capture_output=True, text=True)
            for path in result.stdout.splitlines():
                if "python3" in path.lower():
                    interpreters.append(path.strip())
        except Exception as e:
            print("Error finding interpreters:", e)
        
        interpreters = list(dict.fromkeys(interpreters))
        self.interpreter_combo.clear()
        self.interpreter_combo.addItems(interpreters)
        if self.settings.python_interpreter in interpreters:
            self.interpreter_combo.setCurrentText(self.settings.python_interpreter)

    def change_interpreter(self, interpreter):
        self.settings.python_interpreter = interpreter
        self.settings.save_settings()
        self.statusBar().showMessage(f"Python interpreter set to: {interpreter}")

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

        toggle_terminal_action = QAction("Toggle Terminal", self)
        toggle_terminal_action.setShortcut("Ctrl+`")
        toggle_terminal_action.triggered.connect(self.toggle_terminal)
        terminal_menu.addAction(toggle_terminal_action)

        settings_menu = self.menuBar().addMenu(" &Settings")
        font_settings_action = QAction("Font Settings", self)
        font_settings_action.triggered.connect(self.show_font_settings)
        settings_menu.addAction(font_settings_action)
        reset_action = QAction("Reset to Defaults", self)
        reset_action.triggered.connect(self.reset_to_defaults)
        settings_menu.addAction(reset_action)

        theme_settings_action = QAction("Theme Settings", self)
        theme_settings_action.triggered.connect(self.show_theme_settings)
        settings_menu.addAction(theme_settings_action)

    # In main.py's show_font_settings
    def show_font_settings(self):
        current_editor = self.editor_tabs.currentWidget()
        if current_editor is None:
            default_font = QFont("Fira Code", 12)
            font, ok = QFontDialog.getFont(default_font, self, "Choose Font")
        else:
            font, ok = QFontDialog.getFont(current_editor.font(), self, "Choose Font", QFontDialog.DontUseNativeDialog)
        if ok:
            for i in range(self.editor_tabs.count()):
                editor = self.editor_tabs.widget(i)
                if isinstance(editor, CodeEditor):
                    editor.setFont(font)
            self.settings.editor_font = font
            self.settings.save_settings()

    # In main.py's show_theme_settings
    def show_theme_settings(self):
        theme, ok = QInputDialog.getItem(self, "Choose Theme", "Select a theme:", ["dark", "light"], 0, False)
        if ok:
            self.settings.theme = theme
            self.settings.save_settings()
            self.apply_theme_to_file_explorer()
            self.apply_theme_to_tabs()
            for i in range(self.editor_tabs.count()):
                editor = self.editor_tabs.widget(i)
                if isinstance(editor, CodeEditor):
                    editor.theme = theme
                    editor.apply_theme()
            self.terminal.theme = theme
            self.terminal.apply_theme()
            self.interpreter_combo.setStyleSheet(f"background-color: {THEMES[theme]['AI_PANEL_BG']}; color: {THEMES[theme]['AI_PANEL_FG']}; border: 1px solid #2d2d30; padding: 5px;")
            if self.ai_dock:
                ai_prompt = self.ai_dock.widget().layout().itemAt(0).widget()
                ai_text = self.ai_dock.widget().layout().itemAt(1).widget()
                ai_prompt.setStyleSheet(f"""
                    background-color: {THEMES[theme]['AI_PANEL_BG']};
                    color: {THEMES[theme]['AI_PANEL_FG']};
                    border: 1px solid #2d2d30;
                    padding: 5px;
                """)
                ai_text.setStyleSheet(f"""
                    background-color: {THEMES[theme]['AI_PANEL_BG']};
                    color: {THEMES[theme]['AI_PANEL_FG']};
                    border: 1px solid #2d2d30;
                """)
            self.statusBar().showMessage(f"Theme changed to: {theme}")

    def _setup_file_explorer(self):
        self.tree_view.setRootIndex(self.file_model.index(QDir.currentPath()))
        self.tree_view.setColumnWidth(0, 250)
        self.terminal.change_directory(QDir.currentPath())

    def _setup_status_bar(self):
        self.statusBar().showMessage("Ready")

    def new_file(self):
        try:     
            with open("Untitled.py", "w", encoding='utf-8') as file:
                file.write("#WRITE YOUR COPY RIGHT HERE")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run code: {e}")
        
        editor = CodeEditor(self, theme=self.settings.theme)
        with open("Untitled.py", 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        self.Current_FileName = "Untitled.py"
        editor.setFont(self.settings.editor_font)
        editor.setPlainText(content)
        
        self.editor_tabs.addTab(editor, os.path.basename('Untitled.py'))
        self.editor_tabs.setCurrentWidget(editor)
        self.terminal.change_directory(os.getcwd())

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
                    os.chdir(folder_path)
                    self.terminal.change_directory(folder_path)
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
            editor = CodeEditor(self, theme=self.settings.theme)
            editor.setFont(self.settings.editor_font)
            editor.setPlainText(content)
            self.editor_tabs.addTab(editor, os.path.basename(path))
            self.editor_tabs.setCurrentWidget(editor)
            self.statusBar().showMessage(f"Opened file: {path}")
            os.chdir(os.path.dirname(path))
            self.terminal.change_directory(os.path.dirname(path))
            self.Current_FileName = path
            print(f"Opened file: {path}, completer exists: {hasattr(editor, 'completer')}")
        except Exception as e:
            print(f"Error opening file: {e}")
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
                    os.chdir(os.path.dirname(file_path))
                    self.terminal.change_directory(os.path.dirname(file_path))
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    def close_tab(self, index):
        self.editor_tabs.removeTab(index)

    def run_code(self):
        current_editor = self.editor_tabs.currentWidget()
        if current_editor:
            try:
                code = current_editor.toPlainText()
                Name = self.Current_FileName
                with open(Name, "w", encoding='utf-8') as file:
                    file.write(code)
                interpreter = self.settings.python_interpreter
                self.terminal.run_command(f'"{interpreter}" "{Name}"')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run code: {e}")

    def open_system_terminal(self):
        os.system("start cmd")

    def show_ai_panel(self, selected_text):
        code_block_css = (
            "background-color: #1e1e1e;"
            f"color: {THEMES[self.settings.theme]['AI_PANEL_FG']};"
            "white-space: pre-wrap;"
            "word-wrap: break-word;"
            "padding: 4px;"
            "border-radius: 3px;"
            "margin: 0;"
            "border: none;"
            "display: block;"
        )

        def build_html(prompt=None, result=None):
            highlighted_selected_text = pygments_highlight(selected_text, code_block_css)

            base_style = (
                "font-family: 'Fira Code'; font-size: 8pt; "
                f"background-color: {THEMES[self.settings.theme]['AI_PANEL_BG']};"
                f"color: {THEMES[self.settings.theme]['AI_PANEL_FG']};"
                "padding: 5px;"
            )

            parts = [
                f"<div style=\"{base_style}\">",
                "  <span style=\"color: cyan; font-weight: bold;\">[SELECTED TEXT]:</span><br>",
                highlighted_selected_text
            ]

            if prompt is None:
                parts += [
                    "  <br>",
                    "  <span>Enter your prompt:</span>"
                ]
            else:
                esc_prompt = html.escape(prompt)
                esc_prompt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', esc_prompt)  # Apply bold formatting
                # Prompt Section
                parts += [
                    "  <br>",
                    f"  <div style=\"padding: 4px; border-radius: 3px; background-color: {THEMES[self.settings.theme]['AI_PANEL_BG']}; color: {THEMES[self.settings.theme]['AI_PANEL_FG']};\">",
                    "    <span style=\"color: #ce9178; font-weight: bold;\">[YOUR PROMPT]:</span><br>",
                    f"    <div style=\"white-space: pre-wrap; word-wrap: break-word; margin-top: 4px;\">{esc_prompt}</div>",
                    "  </div>",
                    "  <br>"
                ]
                # AI Section (single)
                raw_result = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', str(result))  # Apply bold formatting to result
                processed_res = process_codeenv(raw_result, code_block_css)
                parts += [
                    f"  <div style=\"padding: 4px; border-radius: 3px; background-color: {THEMES[self.settings.theme]['AI_PANEL_BG']}; color: {THEMES[self.settings.theme]['AI_PANEL_FG']};\">",
                    "    <span style=\"color: #4ec9b0; font-weight: bold;\">[AI]:</span><br>",
                    f"    <div style=\"white-space: pre-wrap; word-wrap: break-word; margin-top: 4px;\">{processed_res}</div>",
                    "  </div>"
                ]

            parts.append("</div>")
            return "\n".join(parts)

        if self.ai_dock is None:
            self.ai_dock = QDockWidget("AI Assistant", self.window())
            self.ai_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
            ai_widget   = QWidget()
            ai_layout   = QVBoxLayout(ai_widget)
            ai_layout.setContentsMargins(0, 0, 0, 0)
            ai_layout.setSpacing(5)

            ai_prompt = QLineEdit()
            ai_prompt.setPlaceholderText("Enter your AI prompt here…")
            ai_prompt.setFont(QFont("Fira Code", 8))
            ai_prompt.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {THEMES[self.settings.theme]['AI_PANEL_BG']};
                    color: {THEMES[self.settings.theme]['AI_PANEL_FG']};
                    border: 1px solid #2d2d30;
                    padding: 5px;
                }}
                QLineEdit:focus {{
                    border: 1px solid #007acc;
                }}
            """)

            ai_text = QTextEdit()
            ai_text.setReadOnly(True)
            ai_text.setFont(QFont("Fira Code", 8))
            ai_text.setAcceptRichText(True)
            ai_text.setContentsMargins(0,0,0,0)

            ai_text.setStyleSheet(f"""
                QTextEdit {{
                    background-color: {THEMES[self.settings.theme]['AI_PANEL_BG']};
                    color: {THEMES[self.settings.theme]['AI_PANEL_FG']};
                    border: 1px solid #2d2d30;
                }}
            """)

            ai_text.setHtml(build_html())

            def submit_prompt():
                prompt = ai_prompt.text().strip()
                if not prompt:
                    self.window().statusBar().showMessage("Prompt cannot be empty", 2000)
                    return

                self.window().statusBar().showMessage("Thinking...", 0)
                try:
                    payload = {"text": f"[SELECTED TEXT]:\n```\n{selected_text}\n```\n\n[PROMPT]: {prompt}"}

                    response = requests.post(
                        "https://gpu-handler.onrender.com/predict",
                        json=payload
                    )
                    response.raise_for_status()

                    result_data = response.json()
                    result = result_data

                except requests.exceptions.RequestException:
                    result = f"Failed to connect to online GPU or API error"
                except Exception:
                    result = f"An unexpected error occurred"

                ai_text.setHtml(build_html(prompt, result))
                ai_prompt.clear()
                self.window().statusBar().showMessage("AI response received", 3000)

            ai_prompt.returnPressed.connect(submit_prompt)

            ai_layout.addWidget(ai_prompt)
            ai_layout.addWidget(ai_text)
            ai_widget.setLayout(ai_layout)
            self.ai_dock.setWidget(ai_widget)
            self.window().addDockWidget(Qt.RightDockWidgetArea, self.ai_dock)

        else:
            ai_text = self.ai_dock.widget().layout().itemAt(1).widget()
            ai_text.setHtml(build_html())

            self.ai_dock.show()

        self.window().statusBar().showMessage("AI panel opened")
        







class MyMonokai(MonokaiStyle):
    styles = MonokaiStyle.styles.copy()
    styles.update({
        Comment:        '#6A9955',  # green
        Name.Function:  'bold   #DCCB7A',  # yellow
        Name.Class:     '#FFFF00',         # yellow
        Keyword:        'bold #DCAADC',
        Token:          '#4EC9B0',
    })
    background_color = "#000000"  # Set background to black explicitly

import re
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters.html import HtmlFormatter


def pygments_highlight(code: str, default_color_css: str) -> str:

    if not code.endswith("\n"):
        code += "\n"
    formatter = HtmlFormatter(noclasses=True, style=MyMonokai, nowrap=True)
    highlighted = highlight(code, PythonLexer(), formatter)

    highlighted = re.sub(r'(</span>)(\s*)(<span class="c">)', r'\1\2\3', highlighted)

    return f'<pre style="{default_color_css}">{highlighted}</pre>'

def process_codeenv(text: str, default_css: str) -> str:
        out = []
        cursor = 0
        for match in re.finditer(r'<CodeEnv>', text):
            start = match.start()
            end_tag = re.search(r'</CodeEnv>', text[start:])
            if end_tag:
                end = start + end_tag.end()
                code = text[start + len('<CodeEnv>'): end - len('</CodeEnv>')]
                out.append(text[cursor:start])  # No escaping yet
                out.append(pygments_highlight(code, default_css))
                cursor = end
            else:
                out.append(text[cursor:start])
                code = text[start + len('<CodeEnv>') :]
                out.append(pygments_highlight(code, default_css))
                cursor = len(text)
                break
        out.append(text[cursor:])
        processed_text = ''.join(out)

        # Escape everything except <b> tags and highlighted code
        def escape_except_html(match):
            return html.escape(match.group(0))

        # First, protect all <b> tags and highlighted code blocks from escaping
        processed_text = re.sub(r'<(?!/?(b|span|div)).*?>', escape_except_html, processed_text)
        
        # Bold formatting for **text** (remove asterisks and apply bold)
        processed_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', processed_text)

        return processed_text
