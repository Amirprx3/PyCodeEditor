
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit
from PyQt5.QtCore import QProcess, Qt, QSize
from PyQt5.QtGui import QFont

class Terminal(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_output)
        self.process.readyReadStandardError.connect(self.handle_error)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 12))  # Set Font & Size for terminal output

        self.input = QLineEdit()
        self.input.returnPressed.connect(self.execute_command)
        self.input.setFont(QFont("Consolas", 12))  # Set Font & Size for terminal input

        layout.addWidget(self.output)
        layout.addWidget(self.input)


    def execute_command(self):
        command = self.input.text()
        self.input.clear()

        # Check for clear/cls command
        if command.lower() in ["clear", "cls"]:
            self.output.clear()  # Clear the terminal output
            return

        self.output.appendHtml(f"<b>{command}</b>")
        self.run_command(command)

    def handle_output(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.output.appendPlainText(data)

    def handle_error(self):
        data = self.process.readAllStandardError().data().decode()
        self.output.appendPlainText(data)

    def run_command(self, command):
        # Run the command directly in Windows Command Prompt
        self.process.start("cmd.exe", ["/C", command])

    def sizeHint(self):
        return QSize(2, 2)
