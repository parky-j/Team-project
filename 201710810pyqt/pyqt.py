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
    win.setGeometry(0, 0, 1920, 1080)
    win.setWindowTitle("Stroke counting")
    win.setWindowIcon(QIcon('iconimg.jpg'))
    #btn= QPushButton('Quit')
    #btn.move(50,50)
    #btn.resize(btn.sizeHint())
    #btn.clicked.connect(QCoreApplication.instance().quit)
    #btn.setWindowTitle('Quit Button')

    label=QtWidgets.QLabel(win)
    label.setText("Stroke Counting")
    label.move(930,50)
    btn = QPushButton('Quit', win)
    btn.move(1800, 1000)
    btn.resize(btn.sizeHint())
    btn.setShortcut('Ctrl+Q')
    btn.clicked.connect(QCoreApplication.instance().quit)
    QToolTip.setFont(QFont('SanSerif', 10))
    btn.setToolTip('This is a quit button')
   
    exp = QPushButton('File explore', win)
    exp.clicked.connect(OpenFile)
    exp.move(30, 15)
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