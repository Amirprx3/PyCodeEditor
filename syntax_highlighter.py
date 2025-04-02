
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
            'function_keyword': QColor('#DCAADC'),  
            'function_name': QColor('#DCCB7A'),  
            'class': QColor('#4EC9B0'),
        }
        self._rules = []
        self._setup_rules()

    def _setup_rules(self):
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'del',
            'elif', 'else', 'except', 'False', 'finally', 'for', 'from', 'global',
            'if', 'import', 'in', 'is', 'lambda', 'None', 'nonlocal', 'not', 'or',
            'pass', 'print', 'raise', 'return', 'True', 'try', 'while', 'with', 'yield','self'
        ]
        self._rules.append((QRegularExpression(r'\b(%s)\b' % '|'.join(keywords)), 'keyword'))
        self._rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), 'string'))
        self._rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), 'string'))
        self._rules.append((QRegularExpression(r'#.*'), 'comment'))
        self._rules.append((QRegularExpression(r'\b\d+\b'), 'number'))
        self._rules.append((QRegularExpression(r'\bdef\b'), 'function_keyword'))  # Match "def" separately
        self._rules.append((QRegularExpression(r'\bdef\b\s+(\w+)'), 'function_name'))  # Match function name separately
        self._rules.append((QRegularExpression(r'\bclass\b\s*(\w+)'), 'class'))

    def highlightBlock(self, text):
        for pattern, style in self._rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                if style == 'function_name' and match.lastCapturedIndex() > 0:
                    start = match.capturedStart(1)
                    length = match.capturedLength(1)
                else:
                    start = match.capturedStart()
                    length = match.capturedLength()

                self.setFormat(start, length, self._styles[style])
