from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit
from PyQt5.QtCore import QProcess, Qt, QSize
from PyQt5.QtGui import QFont

class Terminal(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)  # Combining standard output and error
        self.process.readyRead.connect(self.handle_output)  # Read output in real time
        self.process.start("cmd.exe")  # Run cmd running

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 12))  # Adjust the output font and size

        self.input = QLineEdit()
        self.input.returnPressed.connect(self.execute_command)
        self.input.setFont(QFont("Consolas", 12))  # Adjust the input font and size

        layout.addWidget(self.output)
        layout.addWidget(self.input)

    def execute_command(self):
        command = self.input.text()
        self.input.clear()

        if command.lower() in ["clear", "cls"]:
            self.output.clear()  # Clear terminal output
            return

        self.output.appendHtml(f"<b>{command}</b>")  # Show command in output
        self.run_command(command)

    def handle_output(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.output.appendPlainText(data)

    def run_command(self, command):
        if self.process.state() == QProcess.NotRunning:
            self.process.start("cmd.exe")  # If CMD is closed, open it again.

        self.process.write((command + "\n").encode())  # Send command to CMD
        self.process.waitForBytesWritten()

    def sizeHint(self):
        return QSize(2, 2)