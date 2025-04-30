from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import QProcess, Qt, QSize
from PyQt5.QtGui import QFont, QTextCursor
import os
import platform
from settings import THEMES

class Terminal(QWidget):
    def __init__(self, theme="dark"):
        super().__init__()
        self.theme = theme
        self.current_dir = os.getcwd()
        self.process = None
        self._setup_ui()
        self._setup_process()
        self.update_prompt()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 2, 5, 2)
        
        clear_button = QPushButton("Clear")
        clear_button.setFixedWidth(60)
        clear_button.clicked.connect(self.clear)
        clear_button.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {THEMES[self.theme]['AI_PANEL_BG']}; 
                color: {THEMES[self.theme]['AI_PANEL_FG']}; 
                border: 1px solid #2d2d30; 
                padding: 3px; 
            }}
            QPushButton:hover {{ background-color: #4ec9b0; }}
        """)
        
        system_terminal_button = QPushButton("Open System Terminal")
        system_terminal_button.clicked.connect(self.open_system_terminal)
        system_terminal_button.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {THEMES[self.theme]['AI_PANEL_BG']}; 
                color: {THEMES[self.theme]['AI_PANEL_FG']}; 
                border: 1px solid #2d2d30; 
                padding: 3px; 
            }}
            QPushButton:hover {{ background-color: #4ec9b0; }}
        """)
        
        toolbar_layout.addWidget(clear_button)
        toolbar_layout.addWidget(system_terminal_button)
        toolbar_layout.addStretch()
        toolbar.setLayout(toolbar_layout)
        layout.addWidget(toolbar)

        # Terminal output
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Fira Code", 12))
        self.apply_theme()
        layout.addWidget(self.output)

        # Command input
        self.input = QLineEdit()
        self.input.setFont(QFont("Fira Code", 12))
        self.input.returnPressed.connect(self.execute_command)
        self.input.setStyleSheet(f"""
            background-color: {THEMES[self.theme]['TERMINAL_BG']};
            color: {THEMES[self.theme]['TERMINAL_FG']};
            border: 1px solid #2d2d30;
            padding: 5px;
        """)
        layout.addWidget(self.input)

    def apply_theme(self):
        self.output.setStyleSheet(f"""
            background-color: {THEMES[self.theme]['TERMINAL_BG']};
            color: {THEMES[self.theme]['TERMINAL_FG']};
            border: 1px solid #2d2d30;
            padding: 5px;
        """)

    def _setup_process(self):
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyRead.connect(self.handle_output)
        self.process.start("cmd.exe" if platform.system() == "Windows" else "bash")
        self.process.setWorkingDirectory(self.current_dir)

    def update_prompt(self):
        prompt = f"{self.current_dir}> "
        self.output.appendPlainText(prompt)
        self.output.moveCursor(QTextCursor.End)

    def change_directory(self, path):
        try:
            os.chdir(path)
            self.current_dir = os.getcwd()
            self.process.setWorkingDirectory(self.current_dir)
            self.output.appendPlainText(f"cd {path}")
            self.update_prompt()
        except Exception as e:
            self.output.appendPlainText(f"Error changing directory: {e}")
            self.update_prompt()

    def execute_command(self):
        command = self.input.text().strip()
        self.input.clear()
        if not command:
            self.update_prompt()
            return

        self.output.appendHtml(f"<b>{command}</b>")
        if command.lower() in ["clear", "cls"]:
            self.clear()
            return
        elif command.startswith("cd "):
            path = command[3:].strip()
            if path:
                self.change_directory(path)
            return
        elif command in ("dir", "ls"):
            command = "dir" if platform.system() == "Windows" else "ls"

        self.run_command(command)

    def handle_output(self):
        data = self.process.readAllStandardOutput().data().decode(errors="ignore")
        self.output.appendPlainText(data.strip())
        self.output.moveCursor(QTextCursor.End)

    def run_command(self, command):
        if self.process.state() == QProcess.NotRunning:
            self._setup_process()
        self.process.write((command + "\n").encode())
        self.process.waitForBytesWritten()

    def clear(self):
        self.output.clear()
        self.update_prompt()

    def open_system_terminal(self):
        if platform.system() == "Windows":
            os.system(f"start cmd /K cd {self.current_dir}")
        else:
            os.system(f"x-terminal-emulator --working-directory={self.current_dir}")

    def sizeHint(self):
        return QSize(2, 2)