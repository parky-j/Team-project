from fnmatch import fnmatchcase
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTextEdit, QFileDialog, QPushButton, QToolTip, QAction, qApp, QLabel, QVBoxLayout
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QCoreApplication

import sys

from setuptools import find_namespace_packages



def demo():
    app=QApplication(sys.argv)
    win= QMainWindow()
    textEdit = QTextEdit()
    win.setCentralWidget(textEdit)
    win.statusBar()
    win.statusBar().showMessage('Ready')
    win.setGeometry(300, 300, 300, 300)
    win.setWindowTitle("Demo")
    win.setWindowIcon(QIcon('iconimg.jpg'))
    #btn= QPushButton('Quit')
    #btn.move(50,50)
    #btn.resize(btn.sizeHint())
    #btn.clicked.connect(QCoreApplication.instance().quit)
    #btn.setWindowTitle('Quit Button')

    label=QtWidgets.QLabel(win)
    label.setText("Rock")
    label.move(100,100)
    btn = QPushButton('Quit', win)
    btn.move(150, 150)
    btn.resize(btn.sizeHint())
    btn.setShortcut('Ctrl+Q')
    btn.clicked.connect(QCoreApplication.instance().quit)
    QToolTip.setFont(QFont('SanSerif', 10))
    btn.setToolTip('This is a quit button')
   
    exp = QPushButton('File explore', win)
    exp.clicked.connect(OpenFile)
    #exp = QFileDialog.getOpenFileName(win,'Open File', './')
    exp.setShortcut('Ctrl+E')
    exp.setStatusTip('Open new File')
    label1 = QLabel()

    layout = QVBoxLayout()
    layout.addWidget(exp)
    layout.addWidget(label1)
    win.setLayout(layout)

    win.show()
    sys.exit(app.exec_())
def OpenFile():
    fname = QFileDialog.getOpenFileName()
    fname.label1.setText(fname[0])
demo()