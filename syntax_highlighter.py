from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from PyQt5.QtCore import QRegularExpression


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self._styles = {
            'keyword': QColor('#569CD6'),
            'string': QColor('#CE9178'),
            'comment': QColor('#6A9955'),
            'number': QColor('#B5CEA8'),
            'function': QColor('#DCAADC'),
            'class': QColor('#4EC9B0'),
        }
        self._rules = []
        self._setup_rules()

    def _setup_rules(self):
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'False', 'finally', 'for', 'from', 'global',
            'if', 'import', 'in', 'is', 'lambda', 'None', 'nonlocal', 'not', 'or',
            'pass', 'print', 'raise', 'return', 'True', 'try', 'while', 'with', 'yield'
        ]
        self._rules.append((QRegularExpression(r'\b(%s)\b' % '|'.join(keywords)), 'keyword'))
        self._rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), 'string'))
        self._rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), 'string'))
        self._rules.append((QRegularExpression(r'#.*'), 'comment'))
        self._rules.append((QRegularExpression(r'\b\d+\b'), 'number'))
        self._rules.append((QRegularExpression(r'\bdef\b\s*(\w+)'), 'function'))
        self._rules.append((QRegularExpression(r'\bclass\b\s*(\w+)'), 'class'))

    def highlightBlock(self, text):
        for pattern, style in self._rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(
                    match.capturedStart(),
                    match.capturedLength(),
                    self._styles[style]
                )