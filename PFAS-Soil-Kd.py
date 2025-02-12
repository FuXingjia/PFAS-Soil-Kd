import sys
from os.path import exists
from threading import Thread
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QTableWidgetItem, QStyleFactory
from bayes_opt import BayesianOptimization
from joblib import load, dump
from lightgbm import LGBMRegressor
from numpy import array, hstack, zeros, sqrt, around
from padelpy import from_smiles
from pandas import DataFrame, read_excel, read_csv, concat
from rdkit.Chem import Descriptors, MolFromSmiles, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from PyQt5 import QtCore, QtGui, QtWidgets
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(919, 677)
        font = QtGui.QFont()
        font.setItalic(False)
        font.setUnderline(False)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.sideBar = QtWidgets.QWidget(self.centralwidget)
        self.sideBar.setMinimumSize(QtCore.QSize(110, 0))
        self.sideBar.setStyleSheet("background-color: rgb(228, 228, 228);")
        self.sideBar.setObjectName("sideBar")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.sideBar)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Pred_tB = QtWidgets.QToolButton(self.sideBar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Pred_tB.sizePolicy().hasHeightForWidth())
        self.Pred_tB.setSizePolicy(sizePolicy)
        self.Pred_tB.setMinimumSize(QtCore.QSize(86, 90))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.Pred_tB.setFont(font)
        self.Pred_tB.setStyleSheet("\n"
"QToolButton{   \n"
"    border-top: 3px outset transparent; \n"
"    border-bottom: 7px outset transparent;\n"
"    border-right: 3px outset transparent;\n"
"    border-left: 3px outset transparent;\n"
"    min-width: 80px;\n"
"    min-height: 80px;\n"
"    background-color: rgb(228, 228, 228);\n"
"    font: 700 13pt \"Times New Roman\";\n"
"}\n"
"\n"
"\n"
"QToolButton:hover{\n"
"    background-color: rgb(205, 205, 205);\n"
"}\n"
"\n"
"\n"
"QToolButton:pressed,QToolButton:checked{\n"
"    border-left: 3px outset rgb(93, 95, 97);\n"
"    background-color: rgb(246, 246, 246);    \n"
"}\n"
"\n"
"QPushButton:default {\n"
"    border-color: navy; /* make the default button prominent */\n"
"}")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/pre.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Pred_tB.setIcon(icon)
        self.Pred_tB.setIconSize(QtCore.QSize(64, 64))
        self.Pred_tB.setCheckable(True)
        self.Pred_tB.setAutoExclusive(True)
        self.Pred_tB.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.Pred_tB.setObjectName("Pred_tB")
        self.verticalLayout.addWidget(self.Pred_tB)
        self.Retrain_tB = QtWidgets.QToolButton(self.sideBar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Retrain_tB.sizePolicy().hasHeightForWidth())
        self.Retrain_tB.setSizePolicy(sizePolicy)
        self.Retrain_tB.setMinimumSize(QtCore.QSize(86, 90))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.Retrain_tB.setFont(font)
        self.Retrain_tB.setStyleSheet("\n"
"QToolButton{   \n"
"    border-top: 3px outset transparent;\n"
"    border-bottom: 7px outset transparent;\n"
"    border-right: 3px outset transparent;\n"
"    border-left: 3px outset transparent;\n"
"    min-width: 80px;\n"
"    min-height: 80px;\n"
"    background-color: rgb(228, 228, 228);\n"
"    font: 700 13pt \"Times New Roman\";\n"
"}\n"
"\n"
"\n"
"QToolButton:hover{\n"
"    background-color: rgb(205, 205, 205);\n"
"}\n"
"\n"
"\n"
"QToolButton:pressed,QToolButton:checked{\n"
"    border-left: 3px outset rgb(93, 95, 97);\n"
"    background-color: rgb(246, 246, 246);    \n"
"}\n"
"\n"
"QPushButton:default {\n"
"    border-color: navy; /* make the default button prominent */\n"
"}")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icon/train.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Retrain_tB.setIcon(icon1)
        self.Retrain_tB.setIconSize(QtCore.QSize(64, 64))
        self.Retrain_tB.setCheckable(True)
        self.Retrain_tB.setAutoExclusive(True)
        self.Retrain_tB.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.Retrain_tB.setObjectName("Retrain_tB")
        self.verticalLayout.addWidget(self.Retrain_tB)
        self.Descrip_tB = QtWidgets.QToolButton(self.sideBar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Descrip_tB.sizePolicy().hasHeightForWidth())
        self.Descrip_tB.setSizePolicy(sizePolicy)
        self.Descrip_tB.setMinimumSize(QtCore.QSize(86, 90))
        self.Descrip_tB.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.Descrip_tB.setFont(font)
        self.Descrip_tB.setStyleSheet("\n"
"QToolButton{   \n"
"    border-top: 3px outset transparent;\n"
"    border-bottom: 7px outset transparent;\n"
"    border-right: 3px outset transparent;\n"
"    border-left: 3px outset transparent;\n"
"    min-width: 80px;\n"
"    min-height: 80px;\n"
"    background-color: rgb(228, 228, 228);\n"
"    font: 700 13pt \"Times New Roman\";\n"
"}\n"
"\n"
"\n"
"QToolButton:hover{\n"
"    background-color: rgb(205, 205, 205);\n"
"}\n"
"\n"
"\n"
"QToolButton:pressed,QToolButton:checked{\n"
"    border-left: 3px outset rgb(93, 95, 97);\n"
"    background-color: rgb(246, 246, 246);    \n"
"}\n"
"\n"
"QPushButton:default {\n"
"    border-color: navy; /* make the default button prominent */\n"
"}")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icon/MD.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Descrip_tB.setIcon(icon2)
        self.Descrip_tB.setIconSize(QtCore.QSize(64, 64))
        self.Descrip_tB.setCheckable(True)
        self.Descrip_tB.setAutoExclusive(True)
        self.Descrip_tB.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.Descrip_tB.setObjectName("Descrip_tB")
        self.verticalLayout.addWidget(self.Descrip_tB)
        spacerItem = QtWidgets.QSpacerItem(50, 200, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout.addWidget(self.sideBar, 0, 0, 1, 1)
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setItalic(False)
        font.setUnderline(False)
        self.stackedWidget.setFont(font)
        self.stackedWidget.setStyleSheet("")
        self.stackedWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.stackedWidget.setFrameShadow(QtWidgets.QFrame.Plain)
        self.stackedWidget.setLineWidth(3)
        self.stackedWidget.setMidLineWidth(0)
        self.stackedWidget.setObjectName("stackedWidget")
        self.PredWindow = QtWidgets.QWidget()
        font = QtGui.QFont()
        self.PredWindow.setFont(font)
        self.PredWindow.setStyleSheet("border-color: rgb(0, 255, 255);")
        self.PredWindow.setObjectName("PredWindow")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.PredWindow)
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.gridLayout_2.setContentsMargins(20, 12, 20, 10)
        self.gridLayout_2.setHorizontalSpacing(5)
        self.gridLayout_2.setVerticalSpacing(12)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.Clear_Button = QtWidgets.QPushButton(self.PredWindow)
        self.Clear_Button.setMinimumSize(QtCore.QSize(0, 25))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Clear_Button.setFont(font)
        self.Clear_Button.setStyleSheet("background-color: rgb(255, 0, 0);\n"
"font: 12pt \"Times New Roman\";")
        self.Clear_Button.setObjectName("Clear_Button")
        self.gridLayout_2.addWidget(self.Clear_Button, 3, 8, 1, 2)
        self.Pred_line_1 = QtWidgets.QFrame(self.PredWindow)
        self.Pred_line_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Pred_line_1.setLineWidth(2)
        self.Pred_line_1.setFrameShape(QtWidgets.QFrame.HLine)
        self.Pred_line_1.setObjectName("Pred_line_1")
        self.gridLayout_2.addWidget(self.Pred_line_1, 0, 0, 1, 10)
        self.Data_output_Button = QtWidgets.QToolButton(self.PredWindow)
        self.Data_output_Button.setMinimumSize(QtCore.QSize(25, 25))
        font = QtGui.QFont()
        self.Data_output_Button.setFont(font)
        self.Data_output_Button.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.Data_output_Button.setObjectName("Data_output_Button")
        self.gridLayout_2.addWidget(self.Data_output_Button, 7, 9, 1, 1)
        self.tableWidget = QtWidgets.QTableWidget(self.PredWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMinimumSize(QtCore.QSize(0, 200))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        self.tableWidget.setFont(font)
        self.tableWidget.setAutoFillBackground(False)
        self.tableWidget.setStyleSheet("font: 9pt \"Times New Roman\";")
        self.tableWidget.setDragEnabled(True)
        self.tableWidget.setDragDropOverwriteMode(True)
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setCornerButtonEnabled(True)
        self.tableWidget.setRowCount(100)
        self.tableWidget.setColumnCount(100)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setDefaultSectionSize(40)
        self.tableWidget.verticalHeader().setDefaultSectionSize(20)
        self.tableWidget.verticalHeader().setMinimumSectionSize(20)
        self.gridLayout_2.addWidget(self.tableWidget, 6, 0, 1, 10)
        self.Model_choose_Box = QtWidgets.QComboBox(self.PredWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Model_choose_Box.sizePolicy().hasHeightForWidth())
        self.Model_choose_Box.setSizePolicy(sizePolicy)
        self.Model_choose_Box.setMinimumSize(QtCore.QSize(0, 0))
        self.Model_choose_Box.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Model_choose_Box.setFont(font)
        self.Model_choose_Box.setStyleSheet("background-color: rgb(255, 170, 127);\n"
"font: 12pt \"Times New Roman\";")
        self.Model_choose_Box.setEditable(False)
        self.Model_choose_Box.setDuplicatesEnabled(False)
        self.Model_choose_Box.setObjectName("Model_choose_Box")
        self.Model_choose_Box.addItem("")
        self.Model_choose_Box.addItem("")
        self.gridLayout_2.addWidget(self.Model_choose_Box, 1, 2, 1, 8)
        self.predict_text = QtWidgets.QTextBrowser(self.PredWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predict_text.sizePolicy().hasHeightForWidth())
        self.predict_text.setSizePolicy(sizePolicy)
        self.predict_text.setMinimumSize(QtCore.QSize(0, 25))
        self.predict_text.setMaximumSize(QtCore.QSize(16777215, 35))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.predict_text.setFont(font)
        self.predict_text.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 11pt \"Times New Roman\";")
        self.predict_text.setFrameShadow(QtWidgets.QFrame.Plain)
        self.predict_text.setUndoRedoEnabled(False)
        self.predict_text.setObjectName("predict_text")
        self.gridLayout_2.addWidget(self.predict_text, 4, 0, 1, 10)
        self.Pred_line_2 = QtWidgets.QFrame(self.PredWindow)
        self.Pred_line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Pred_line_2.setLineWidth(2)
        self.Pred_line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.Pred_line_2.setObjectName("Pred_line_2")
        self.gridLayout_2.addWidget(self.Pred_line_2, 9, 0, 1, 10)
        self.Data_input_Button = QtWidgets.QToolButton(self.PredWindow)
        self.Data_input_Button.setMinimumSize(QtCore.QSize(25, 25))
        font = QtGui.QFont()
        self.Data_input_Button.setFont(font)
        self.Data_input_Button.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.Data_input_Button.setObjectName("Data_input_Button")
        self.gridLayout_2.addWidget(self.Data_input_Button, 2, 9, 1, 1)
        self.Data_output_label = QtWidgets.QLabel(self.PredWindow)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        self.Data_output_label.setFont(font)
        self.Data_output_label.setStyleSheet("font: 700 14pt \"Times New Roman\";")
        self.Data_output_label.setObjectName("Data_output_label")
        self.gridLayout_2.addWidget(self.Data_output_label, 7, 0, 1, 1)
        self.Start_Button = QtWidgets.QPushButton(self.PredWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Start_Button.setFont(font)
        self.Start_Button.setStyleSheet("background-color: rgb(170, 255, 127);\n"
"font: 12pt \"Times New Roman\";")
        self.Start_Button.setObjectName("Start_Button")
        self.gridLayout_2.addWidget(self.Start_Button, 3, 0, 1, 8)
        self.Choose_model_label = QtWidgets.QLabel(self.PredWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Choose_model_label.sizePolicy().hasHeightForWidth())
        self.Choose_model_label.setSizePolicy(sizePolicy)
        self.Choose_model_label.setMinimumSize(QtCore.QSize(0, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        self.Choose_model_label.setFont(font)
        self.Choose_model_label.setStyleSheet("font: 700 14pt \"Times New Roman\";")
        self.Choose_model_label.setObjectName("Choose_model_label")
        self.gridLayout_2.addWidget(self.Choose_model_label, 1, 0, 1, 2)
        self.Data_input_Edit = QtWidgets.QLineEdit(self.PredWindow)
        self.Data_input_Edit.setMinimumSize(QtCore.QSize(0, 25))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.Data_input_Edit.setFont(font)
        self.Data_input_Edit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Data_input_Edit.setClearButtonEnabled(True)
        self.Data_input_Edit.setObjectName("Data_input_Edit")
        self.gridLayout_2.addWidget(self.Data_input_Edit, 2, 1, 1, 8)
        self.Data_input_label = QtWidgets.QLabel(self.PredWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Data_input_label.sizePolicy().hasHeightForWidth())
        self.Data_input_label.setSizePolicy(sizePolicy)
        self.Data_input_label.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        self.Data_input_label.setFont(font)
        self.Data_input_label.setStyleSheet("font: 700 14pt \"Times New Roman\";")
        self.Data_input_label.setObjectName("Data_input_label")
        self.gridLayout_2.addWidget(self.Data_input_label, 2, 0, 1, 1)
        self.Data_output_Edit = QtWidgets.QLineEdit(self.PredWindow)
        self.Data_output_Edit.setMinimumSize(QtCore.QSize(0, 25))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.Data_output_Edit.setFont(font)
        self.Data_output_Edit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Data_output_Edit.setObjectName("Data_output_Edit")
        self.gridLayout_2.addWidget(self.Data_output_Edit, 7, 1, 1, 8)
        self.Save_Button = QtWidgets.QPushButton(self.PredWindow)
        self.Save_Button.setMinimumSize(QtCore.QSize(0, 25))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Save_Button.setFont(font)
        self.Save_Button.setStyleSheet("background-color: rgb(85, 170, 255);\n"
"font: 12pt \"Times New Roman\";")
        self.Save_Button.setObjectName("Save_Button")
        self.gridLayout_2.addWidget(self.Save_Button, 8, 0, 1, 3)
        self.Save_text = QtWidgets.QTextBrowser(self.PredWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Save_text.sizePolicy().hasHeightForWidth())
        self.Save_text.setSizePolicy(sizePolicy)
        self.Save_text.setMinimumSize(QtCore.QSize(0, 0))
        self.Save_text.setMaximumSize(QtCore.QSize(6170000, 25))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.Save_text.setFont(font)
        self.Save_text.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Save_text.setObjectName("Save_text")
        self.gridLayout_2.addWidget(self.Save_text, 8, 3, 1, 7)
        self.results_label = QtWidgets.QLabel(self.PredWindow)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        self.results_label.setFont(font)
        self.results_label.setStyleSheet("font: 700 14pt \"Times New Roman\";")
        self.results_label.setObjectName("results_label")
        self.gridLayout_2.addWidget(self.results_label, 5, 0, 1, 8)
        self.gridLayout_2.setRowStretch(0, 1)
        self.stackedWidget.addWidget(self.PredWindow)
        self.ModelWindow = QtWidgets.QWidget()
        self.ModelWindow.setObjectName("ModelWindow")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.ModelWindow)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.Open_trainset_file_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.Open_trainset_file_lineEdit.setMinimumSize(QtCore.QSize(0, 25))
        self.Open_trainset_file_lineEdit.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Open_trainset_file_lineEdit.setInputMask("")
        self.Open_trainset_file_lineEdit.setText("")
        self.Open_trainset_file_lineEdit.setClearButtonEnabled(True)
        self.Open_trainset_file_lineEdit.setObjectName("Open_trainset_file_lineEdit")
        self.gridLayout_4.addWidget(self.Open_trainset_file_lineEdit, 3, 2, 1, 9)
        self.Bayesian_pB = QtWidgets.QPushButton(self.ModelWindow)
        self.Bayesian_pB.setMinimumSize(QtCore.QSize(0, 25))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Bayesian_pB.setFont(font)
        self.Bayesian_pB.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"font: 12pt \"Times New Roman\";")
        self.Bayesian_pB.setObjectName("Bayesian_pB")
        self.gridLayout_4.addWidget(self.Bayesian_pB, 9, 0, 1, 8)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem1, 12, 6, 1, 1)
        self.Model_data_select_label = QtWidgets.QLabel(self.ModelWindow)
        self.Model_data_select_label.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        self.Model_data_select_label.setFont(font)
        self.Model_data_select_label.setStyleSheet("font: 700 14pt \"Times New Roman\";")
        self.Model_data_select_label.setObjectName("Model_data_select_label")
        self.gridLayout_4.addWidget(self.Model_data_select_label, 1, 0, 1, 12)
        self.RMSE_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.RMSE_label.setFont(font)
        self.RMSE_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.RMSE_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.RMSE_label.setObjectName("RMSE_label")
        self.gridLayout_4.addWidget(self.RMSE_label, 18, 3, 1, 1)
        self.Iterations_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Iterations_label.setFont(font)
        self.Iterations_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Iterations_label.setObjectName("Iterations_label")
        self.gridLayout_4.addWidget(self.Iterations_label, 6, 0, 1, 1)
        self.Max_depth_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.Max_depth_lineEdit.setMinimumSize(QtCore.QSize(150, 0))
        self.Max_depth_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Max_depth_lineEdit.setClearButtonEnabled(True)
        self.Max_depth_lineEdit.setObjectName("Max_depth_lineEdit")
        self.gridLayout_4.addWidget(self.Max_depth_lineEdit, 12, 9, 1, 1)
        self.R2_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.R2_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.R2_lineEdit.setInputMask("")
        self.R2_lineEdit.setClearButtonEnabled(True)
        self.R2_lineEdit.setObjectName("R2_lineEdit")
        self.gridLayout_4.addWidget(self.R2_lineEdit, 18, 1, 1, 1)
        self.Start_retraining_pB = QtWidgets.QPushButton(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Start_retraining_pB.setFont(font)
        self.Start_retraining_pB.setStyleSheet("background-color: rgb(170, 255, 127);\n"
"font: 12pt \"Times New Roman\";")
        self.Start_retraining_pB.setObjectName("Start_retraining_pB")
        self.gridLayout_4.addWidget(self.Start_retraining_pB, 16, 0, 1, 11)
        self.Model_retrain_line_1 = QtWidgets.QFrame(self.ModelWindow)
        self.Model_retrain_line_1.setMinimumSize(QtCore.QSize(50, 5))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(False)
        font.setUnderline(False)
        self.Model_retrain_line_1.setFont(font)
        self.Model_retrain_line_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Model_retrain_line_1.setLineWidth(2)
        self.Model_retrain_line_1.setFrameShape(QtWidgets.QFrame.HLine)
        self.Model_retrain_line_1.setObjectName("Model_retrain_line_1")
        self.gridLayout_4.addWidget(self.Model_retrain_line_1, 0, 0, 1, 11)
        self.R2_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.R2_label.setFont(font)
        self.R2_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.R2_label.setAutoFillBackground(False)
        self.R2_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.R2_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.R2_label.setObjectName("R2_label")
        self.gridLayout_4.addWidget(self.R2_label, 18, 0, 1, 1)
        self.Drop_rate_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Drop_rate_label.setFont(font)
        self.Drop_rate_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Drop_rate_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Drop_rate_label.setObjectName("Drop_rate_label")
        self.gridLayout_4.addWidget(self.Drop_rate_label, 14, 0, 1, 1)
        self.Reg_alpha_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Reg_alpha_label.setFont(font)
        self.Reg_alpha_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Reg_alpha_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Reg_alpha_label.setObjectName("Reg_alpha_label")
        self.gridLayout_4.addWidget(self.Reg_alpha_label, 13, 3, 1, 1)
        self.Bayesian_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        self.Bayesian_label.setFont(font)
        self.Bayesian_label.setStyleSheet("font: 700 14pt \"Times New Roman\";")
        self.Bayesian_label.setObjectName("Bayesian_label")
        self.gridLayout_4.addWidget(self.Bayesian_label, 5, 0, 1, 11)
        self.Bayesian_textBrowser = QtWidgets.QTextBrowser(self.ModelWindow)
        self.Bayesian_textBrowser.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Bayesian_textBrowser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Bayesian_textBrowser.setReadOnly(False)
        self.Bayesian_textBrowser.setObjectName("Bayesian_textBrowser")
        self.gridLayout_4.addWidget(self.Bayesian_textBrowser, 10, 0, 1, 11)
        self.Select_model_retrain_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Select_model_retrain_label.setFont(font)
        self.Select_model_retrain_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Select_model_retrain_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Select_model_retrain_label.setObjectName("Select_model_retrain_label")
        self.gridLayout_4.addWidget(self.Select_model_retrain_label, 2, 0, 1, 2)
        self.N_estimat_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.N_estimat_label.setFont(font)
        self.N_estimat_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.N_estimat_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.N_estimat_label.setObjectName("N_estimat_label")
        self.gridLayout_4.addWidget(self.N_estimat_label, 12, 3, 1, 1)
        self.Num_leaves_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.Num_leaves_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Num_leaves_lineEdit.setClearButtonEnabled(True)
        self.Num_leaves_lineEdit.setObjectName("Num_leaves_lineEdit")
        self.gridLayout_4.addWidget(self.Num_leaves_lineEdit, 13, 1, 1, 1)
        self.Optimization_range_TextEdit = QtWidgets.QPlainTextEdit(self.ModelWindow)
        self.Optimization_range_TextEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Optimization_range_TextEdit.setObjectName("Optimization_range_TextEdit")
        self.gridLayout_4.addWidget(self.Optimization_range_TextEdit, 8, 0, 1, 11)
        self.Model_parameter_set_label = QtWidgets.QLabel(self.ModelWindow)
        self.Model_parameter_set_label.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        self.Model_parameter_set_label.setFont(font)
        self.Model_parameter_set_label.setStyleSheet("font: 700 14pt \"Times New Roman\";")
        self.Model_parameter_set_label.setObjectName("Model_parameter_set_label")
        self.gridLayout_4.addWidget(self.Model_parameter_set_label, 11, 0, 1, 11)
        self.Model_retrain_line_2 = QtWidgets.QFrame(self.ModelWindow)
        self.Model_retrain_line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Model_retrain_line_2.setLineWidth(2)
        self.Model_retrain_line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.Model_retrain_line_2.setObjectName("Model_retrain_line_2")
        self.gridLayout_4.addWidget(self.Model_retrain_line_2, 20, 0, 1, 11)
        self.Max_drop_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        font.setItalic(False)
        font.setUnderline(False)
        self.Max_drop_lineEdit.setFont(font)
        self.Max_drop_lineEdit.setObjectName("Max_drop_lineEdit")
        self.gridLayout_4.addWidget(self.Max_drop_lineEdit, 14, 4, 1, 2)
        self.Iterations_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.Iterations_lineEdit.setMinimumSize(QtCore.QSize(0, 25))
        self.Iterations_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Iterations_lineEdit.setClearButtonEnabled(True)
        self.Iterations_lineEdit.setObjectName("Iterations_lineEdit")
        self.gridLayout_4.addWidget(self.Iterations_lineEdit, 6, 1, 1, 2)
        self.MAE_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.MAE_label.setFont(font)
        self.MAE_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.MAE_label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.MAE_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.MAE_label.setObjectName("MAE_label")
        self.gridLayout_4.addWidget(self.MAE_label, 18, 7, 1, 1)
        self.Reg_lambda_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.Reg_lambda_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Reg_lambda_lineEdit.setClearButtonEnabled(True)
        self.Reg_lambda_lineEdit.setObjectName("Reg_lambda_lineEdit")
        self.gridLayout_4.addWidget(self.Reg_lambda_lineEdit, 13, 9, 1, 1)
        self.N_estimat_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.N_estimat_lineEdit.setMinimumSize(QtCore.QSize(150, 0))
        self.N_estimat_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.N_estimat_lineEdit.setClearButtonEnabled(True)
        self.N_estimat_lineEdit.setObjectName("N_estimat_lineEdit")
        self.gridLayout_4.addWidget(self.N_estimat_lineEdit, 12, 4, 1, 2)
        self.Max_depth_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Max_depth_label.setFont(font)
        self.Max_depth_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Max_depth_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Max_depth_label.setObjectName("Max_depth_label")
        self.gridLayout_4.addWidget(self.Max_depth_label, 12, 7, 1, 1)
        self.Optimization_range_label = QtWidgets.QLabel(self.ModelWindow)
        self.Optimization_range_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Optimization_range_label.setObjectName("Optimization_range_label")
        self.gridLayout_4.addWidget(self.Optimization_range_label, 7, 0, 1, 11)
        self.Learning_rate_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.Learning_rate_lineEdit.setMinimumSize(QtCore.QSize(150, 0))
        self.Learning_rate_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Learning_rate_lineEdit.setClearButtonEnabled(True)
        self.Learning_rate_lineEdit.setObjectName("Learning_rate_lineEdit")
        self.gridLayout_4.addWidget(self.Learning_rate_lineEdit, 12, 1, 1, 1)
        self.Drop_rate_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.Drop_rate_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Drop_rate_lineEdit.setClearButtonEnabled(True)
        self.Drop_rate_lineEdit.setObjectName("Drop_rate_lineEdit")
        self.gridLayout_4.addWidget(self.Drop_rate_lineEdit, 14, 1, 1, 1)
        self.RMSE_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.RMSE_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.RMSE_lineEdit.setInputMask("")
        self.RMSE_lineEdit.setClearButtonEnabled(True)
        self.RMSE_lineEdit.setObjectName("RMSE_lineEdit")
        self.gridLayout_4.addWidget(self.RMSE_lineEdit, 18, 4, 1, 2)
        self.Retraining_result_label = QtWidgets.QLabel(self.ModelWindow)
        self.Retraining_result_label.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        self.Retraining_result_label.setFont(font)
        self.Retraining_result_label.setStyleSheet("font: 700 14pt \"Times New Roman\";")
        self.Retraining_result_label.setObjectName("Retraining_result_label")
        self.gridLayout_4.addWidget(self.Retraining_result_label, 17, 0, 1, 2)
        self.Max_drop_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(False)
        font.setUnderline(False)
        self.Max_drop_label.setFont(font)
        self.Max_drop_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Max_drop_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Max_drop_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Max_drop_label.setObjectName("Max_drop_label")
        self.gridLayout_4.addWidget(self.Max_drop_label, 14, 3, 1, 1)
        self.Open_retrainset_file_pB = QtWidgets.QPushButton(self.ModelWindow)
        self.Open_retrainset_file_pB.setMinimumSize(QtCore.QSize(0, 25))
        self.Open_retrainset_file_pB.setMaximumSize(QtCore.QSize(1677215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Open_retrainset_file_pB.setFont(font)
        self.Open_retrainset_file_pB.setStyleSheet("background-color: rgb(255, 255, 127);\n"
"font: 12pt \"Times New Roman\";")
        self.Open_retrainset_file_pB.setObjectName("Open_retrainset_file_pB")
        self.gridLayout_4.addWidget(self.Open_retrainset_file_pB, 3, 0, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem2, 12, 2, 1, 1)
        self.MAE_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.MAE_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.MAE_lineEdit.setInputMask("")
        self.MAE_lineEdit.setReadOnly(False)
        self.MAE_lineEdit.setClearButtonEnabled(True)
        self.MAE_lineEdit.setObjectName("MAE_lineEdit")
        self.gridLayout_4.addWidget(self.MAE_lineEdit, 18, 9, 1, 1)
        self.Stop_pB = QtWidgets.QPushButton(self.ModelWindow)
        self.Stop_pB.setMinimumSize(QtCore.QSize(0, 25))
        self.Stop_pB.setStyleSheet("background-color: rgb(255, 0, 0);\n"
"font: 12pt \"Times New Roman\";")
        self.Stop_pB.setObjectName("Stop_pB")
        self.gridLayout_4.addWidget(self.Stop_pB, 9, 9, 1, 2)
        self.Reg_lambda_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Reg_lambda_label.setFont(font)
        self.Reg_lambda_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Reg_lambda_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Reg_lambda_label.setObjectName("Reg_lambda_label")
        self.gridLayout_4.addWidget(self.Reg_lambda_label, 13, 7, 1, 1)
        self.Num_leaves_label = QtWidgets.QLabel(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Num_leaves_label.setFont(font)
        self.Num_leaves_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Num_leaves_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Num_leaves_label.setObjectName("Num_leaves_label")
        self.gridLayout_4.addWidget(self.Num_leaves_label, 13, 0, 1, 1)
        self.Learning_rate_label = QtWidgets.QLabel(self.ModelWindow)
        self.Learning_rate_label.setMaximumSize(QtCore.QSize(16772215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Learning_rate_label.setFont(font)
        self.Learning_rate_label.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.Learning_rate_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Learning_rate_label.setObjectName("Learning_rate_label")
        self.gridLayout_4.addWidget(self.Learning_rate_label, 12, 0, 1, 1)
        self.Model_choose_Box_2 = QtWidgets.QComboBox(self.ModelWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Model_choose_Box_2.sizePolicy().hasHeightForWidth())
        self.Model_choose_Box_2.setSizePolicy(sizePolicy)
        self.Model_choose_Box_2.setMinimumSize(QtCore.QSize(0, 25))
        self.Model_choose_Box_2.setMaximumSize(QtCore.QSize(6170000, 25))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Model_choose_Box_2.setFont(font)
        self.Model_choose_Box_2.setStyleSheet("background-color: rgb(255, 170, 127);\n"
"font: 12pt \"Times New Roman\";")
        self.Model_choose_Box_2.setEditable(False)
        self.Model_choose_Box_2.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.Model_choose_Box_2.setDuplicatesEnabled(False)
        self.Model_choose_Box_2.setObjectName("Model_choose_Box_2")
        self.Model_choose_Box_2.addItem("")
        self.Model_choose_Box_2.addItem("")
        self.gridLayout_4.addWidget(self.Model_choose_Box_2, 2, 2, 1, 9)
        self.Reg_alpha_lineEdit = QtWidgets.QLineEdit(self.ModelWindow)
        self.Reg_alpha_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Reg_alpha_lineEdit.setClearButtonEnabled(True)
        self.Reg_alpha_lineEdit.setObjectName("Reg_alpha_lineEdit")
        self.gridLayout_4.addWidget(self.Reg_alpha_lineEdit, 13, 4, 1, 2)
        self.checkBox = QtWidgets.QCheckBox(self.ModelWindow)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.checkBox.setFont(font)
        self.checkBox.setStyleSheet("font: 700 13pt \"Times New Roman\";")
        self.checkBox.setChecked(False)
        self.checkBox.setTristate(False)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_4.addWidget(self.checkBox, 14, 7, 1, 3)
        self.stackedWidget.addWidget(self.ModelWindow)
        self.DescripWindow = QtWidgets.QWidget()
        self.DescripWindow.setObjectName("DescripWindow")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.DescripWindow)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.Export_data_pB = QtWidgets.QPushButton(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.Export_data_pB.setFont(font)
        self.Export_data_pB.setStyleSheet("background-color: rgb(85, 170, 255);\n"
"font: 11pt \"Times New Roman\";")
        self.Export_data_pB.setObjectName("Export_data_pB")
        self.gridLayout_3.addWidget(self.Export_data_pB, 10, 7, 1, 2)
        self.MD_line_2 = QtWidgets.QFrame(self.DescripWindow)
        self.MD_line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.MD_line_2.setLineWidth(2)
        self.MD_line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.MD_line_2.setObjectName("MD_line_2")
        self.gridLayout_3.addWidget(self.MD_line_2, 14, 1, 1, 8)
        self.Batch_calcu_lineEdit = QtWidgets.QLineEdit(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.Batch_calcu_lineEdit.setFont(font)
        self.Batch_calcu_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Batch_calcu_lineEdit.setInputMask("")
        self.Batch_calcu_lineEdit.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.Batch_calcu_lineEdit.setCursorPosition(13)
        self.Batch_calcu_lineEdit.setCursorMoveStyle(QtCore.Qt.LogicalMoveStyle)
        self.Batch_calcu_lineEdit.setClearButtonEnabled(True)
        self.Batch_calcu_lineEdit.setObjectName("Batch_calcu_lineEdit")
        self.gridLayout_3.addWidget(self.Batch_calcu_lineEdit, 6, 1, 1, 7)
        self.MD_result_label = QtWidgets.QLabel(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.MD_result_label.setFont(font)
        self.MD_result_label.setStyleSheet("font: 700 13pt \"Times New Roman\";")
        self.MD_result_label.setObjectName("MD_result_label")
        self.gridLayout_3.addWidget(self.MD_result_label, 10, 1, 1, 1)
        self.Single_calcu_pB = QtWidgets.QPushButton(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.Single_calcu_pB.setFont(font)
        self.Single_calcu_pB.setStyleSheet("background-color: rgb(170, 255, 127);\n"
"font: 11pt \"Times New Roman\";")
        self.Single_calcu_pB.setObjectName("Single_calcu_pB")
        self.gridLayout_3.addWidget(self.Single_calcu_pB, 4, 7, 1, 2)
        self.Choose_MD_cB = QtWidgets.QComboBox(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Choose_MD_cB.setFont(font)
        self.Choose_MD_cB.setStyleSheet("background-color: rgb(255, 170, 127);\n"
"font: 12pt \"Times New Roman\";")
        self.Choose_MD_cB.setObjectName("Choose_MD_cB")
        self.Choose_MD_cB.addItem("")
        self.Choose_MD_cB.addItem("")
        self.gridLayout_3.addWidget(self.Choose_MD_cB, 1, 3, 1, 6)
        self.Draw_smile_pB = QtWidgets.QPushButton(self.DescripWindow)
        self.Draw_smile_pB.setStyleSheet("font: 11pt \"Times New Roman\";\n"
"background-color: rgb(170, 255, 255);")
        self.Draw_smile_pB.setObjectName("Draw_smile_pB")
        self.gridLayout_3.addWidget(self.Draw_smile_pB, 4, 6, 1, 1)
        self.MD_result_table = QtWidgets.QTableWidget(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        self.MD_result_table.setFont(font)
        self.MD_result_table.setStyleSheet("font: 9pt \"Times New Roman\";")
        self.MD_result_table.setAutoScrollMargin(16)
        self.MD_result_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.MD_result_table.setDefaultDropAction(QtCore.Qt.IgnoreAction)
        self.MD_result_table.setAlternatingRowColors(True)
        self.MD_result_table.setGridStyle(QtCore.Qt.SolidLine)
        self.MD_result_table.setRowCount(100)
        self.MD_result_table.setColumnCount(100)
        self.MD_result_table.setObjectName("MD_result_table")
        self.MD_result_table.horizontalHeader().setDefaultSectionSize(40)
        self.MD_result_table.horizontalHeader().setMinimumSectionSize(30)
        self.MD_result_table.verticalHeader().setDefaultSectionSize(20)
        self.MD_result_table.verticalHeader().setMinimumSectionSize(20)
        self.gridLayout_3.addWidget(self.MD_result_table, 12, 1, 1, 8)
        self.Batch_calcu_tB = QtWidgets.QToolButton(self.DescripWindow)
        self.Batch_calcu_tB.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.Batch_calcu_tB.setObjectName("Batch_calcu_tB")
        self.gridLayout_3.addWidget(self.Batch_calcu_tB, 6, 8, 1, 1)
        self.Choose_MD_label = QtWidgets.QLabel(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.Choose_MD_label.setFont(font)
        self.Choose_MD_label.setStyleSheet("font: 700 13pt \"Times New Roman\";")
        self.Choose_MD_label.setObjectName("Choose_MD_label")
        self.gridLayout_3.addWidget(self.Choose_MD_label, 1, 1, 1, 2)
        self.Single_calcu_lineEdit = QtWidgets.QLineEdit(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.Single_calcu_lineEdit.setFont(font)
        self.Single_calcu_lineEdit.setStyleSheet("font: 11pt \"Times New Roman\";")
        self.Single_calcu_lineEdit.setClearButtonEnabled(True)
        self.Single_calcu_lineEdit.setObjectName("Single_calcu_lineEdit")
        self.gridLayout_3.addWidget(self.Single_calcu_lineEdit, 4, 1, 1, 5)
        self.Single_calcu_label = QtWidgets.QLabel(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.Single_calcu_label.setFont(font)
        self.Single_calcu_label.setStyleSheet("font: 700 13pt \"Times New Roman\";")
        self.Single_calcu_label.setObjectName("Single_calcu_label")
        self.gridLayout_3.addWidget(self.Single_calcu_label, 3, 1, 1, 1)
        self.Batch_calcu_pB = QtWidgets.QPushButton(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.Batch_calcu_pB.setFont(font)
        self.Batch_calcu_pB.setStyleSheet("background-color: rgb(170, 255, 127);\n"
"font: 11pt \"Times New Roman\";")
        self.Batch_calcu_pB.setObjectName("Batch_calcu_pB")
        self.gridLayout_3.addWidget(self.Batch_calcu_pB, 8, 1, 1, 8)
        self.MD_result_progressBar = QtWidgets.QProgressBar(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        self.MD_result_progressBar.setFont(font)
        self.MD_result_progressBar.setStyleSheet("")
        self.MD_result_progressBar.setProperty("value", 0)
        self.MD_result_progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.MD_result_progressBar.setInvertedAppearance(False)
        self.MD_result_progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.MD_result_progressBar.setObjectName("MD_result_progressBar")
        self.gridLayout_3.addWidget(self.MD_result_progressBar, 10, 2, 1, 5)
        self.MD_line_1 = QtWidgets.QFrame(self.DescripWindow)
        self.MD_line_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.MD_line_1.setLineWidth(2)
        self.MD_line_1.setFrameShape(QtWidgets.QFrame.HLine)
        self.MD_line_1.setObjectName("MD_line_1")
        self.gridLayout_3.addWidget(self.MD_line_1, 0, 1, 1, 8)
        self.Batch_calcu_label = QtWidgets.QLabel(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.Batch_calcu_label.setFont(font)
        self.Batch_calcu_label.setStyleSheet("font: 700 13pt \"Times New Roman\";")
        self.Batch_calcu_label.setObjectName("Batch_calcu_label")
        self.gridLayout_3.addWidget(self.Batch_calcu_label, 5, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem3, 3, 2, 1, 7)
        self.Select_quantity_label = QtWidgets.QLabel(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        self.Select_quantity_label.setFont(font)
        self.Select_quantity_label.setStyleSheet("font: 700 13pt \"Times New Roman\";")
        self.Select_quantity_label.setObjectName("Select_quantity_label")
        self.gridLayout_3.addWidget(self.Select_quantity_label, 2, 1, 1, 2)
        self.Select_quantity_cB = QtWidgets.QComboBox(self.DescripWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.Select_quantity_cB.setFont(font)
        self.Select_quantity_cB.setStyleSheet("background-color: rgb(255, 170, 0);\n"
"font: 12pt \"Times New Roman\";")
        self.Select_quantity_cB.setObjectName("Select_quantity_cB")
        self.Select_quantity_cB.addItem("")
        self.Select_quantity_cB.addItem("")
        self.gridLayout_3.addWidget(self.Select_quantity_cB, 2, 3, 1, 6)
        self.stackedWidget.addWidget(self.DescripWindow)
        self.gridLayout.addWidget(self.stackedWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 919, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.action1 = QtWidgets.QAction(MainWindow)
        self.action1.setObjectName("action1")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_as = QtWidgets.QAction(MainWindow)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PFAS-Soil-Kd V1.0"))
        self.Pred_tB.setText(_translate("MainWindow", "Prediction"))
        self.Retrain_tB.setText(_translate("MainWindow", "Model retrain"))
        self.Descrip_tB.setText(_translate("MainWindow", "Descriptors"))
        self.Clear_Button.setText(_translate("MainWindow", "Clear All"))
        self.Data_output_Button.setText(_translate("MainWindow", "..."))
        self.tableWidget.setSortingEnabled(False)
        self.Model_choose_Box.setItemText(0, _translate("MainWindow", "LightGBM trained by PaDEL Dataset"))
        self.Model_choose_Box.setItemText(1, _translate("MainWindow", "LightGBM trained by RDKit Dataset"))
        self.Data_input_Button.setText(_translate("MainWindow", "..."))
        self.Data_output_label.setText(_translate("MainWindow", "Data output:"))
        self.Start_Button.setText(_translate("MainWindow", "Start to predict"))
        self.Choose_model_label.setText(_translate("MainWindow", "Choose a Model:"))
        self.Data_input_label.setText(_translate("MainWindow", "Data input:"))
        self.Save_Button.setText(_translate("MainWindow", "Save the results"))
        self.results_label.setText(_translate("MainWindow", "The results of prediction:"))
        self.Bayesian_pB.setText(_translate("MainWindow", "Bayesian optimization"))
        self.Model_data_select_label.setText(_translate("MainWindow", "Model and data selection"))
        self.RMSE_label.setText(_translate("MainWindow", "RMSE:"))
        self.Iterations_label.setText(_translate("MainWindow", "Iterations:"))
        self.Start_retraining_pB.setText(_translate("MainWindow", "Start retraining"))
        self.R2_label.setText(_translate("MainWindow", "R<sup>2</sup>:"))
        self.Drop_rate_label.setText(_translate("MainWindow", "drop rate:"))
        self.Reg_alpha_label.setText(_translate("MainWindow", "reg alpha:"))
        self.Bayesian_label.setText(_translate("MainWindow", "Bayesian optimization"))
        self.Select_model_retrain_label.setText(_translate("MainWindow", "Select model to retrain:"))
        self.N_estimat_label.setText(_translate("MainWindow", "n estimators:"))
        self.Optimization_range_TextEdit.setPlainText(_translate("MainWindow", "{\'learning_rate\': (0.01,0.5),\n"
"\'n_estimators\': (100,1000),\n"
"\'max_depth\': (3,9),\n"
"\'num_leaves\':(5,255),\n"
"\'reg_lambda\': (0, 10),\n"
"\'reg_alpha\': (0, 10),\n"
"\'drop_rate\': (0.1,0.3),\n"
"\'max_drop\': (20,100)}"))
        self.Model_parameter_set_label.setText(_translate("MainWindow", "Model parameter settings"))
        self.Iterations_lineEdit.setText(_translate("MainWindow", "1500"))
        self.MAE_label.setText(_translate("MainWindow", "MAE:"))
        self.Max_depth_label.setText(_translate("MainWindow", "max depth:"))
        self.Optimization_range_label.setText(_translate("MainWindow", "Parameter optimization range setting:"))
        self.Retraining_result_label.setText(_translate("MainWindow", "Retraining results"))
        self.Max_drop_label.setText(_translate("MainWindow", "max drop:"))
        self.Open_retrainset_file_pB.setText(_translate("MainWindow", "Open trainset file"))
        self.Stop_pB.setText(_translate("MainWindow", "Stop"))
        self.Reg_lambda_label.setText(_translate("MainWindow", "reg lambda:"))
        self.Num_leaves_label.setText(_translate("MainWindow", "num leaves:"))
        self.Learning_rate_label.setText(_translate("MainWindow", "learning rate:"))
        self.Model_choose_Box_2.setItemText(0, _translate("MainWindow", "LightGBM trained by PaDEL Dataset"))
        self.Model_choose_Box_2.setItemText(1, _translate("MainWindow", "LightGBM trained by RDKit Dataset"))
        self.checkBox.setText(_translate("MainWindow", "Save the model after retraining"))
        self.Export_data_pB.setText(_translate("MainWindow", "Export Data"))
        self.Batch_calcu_lineEdit.setText(_translate("MainWindow", "Choose a File"))
        self.MD_result_label.setText(_translate("MainWindow", "Calculation results:"))
        self.Single_calcu_pB.setText(_translate("MainWindow", "Caculated (Single)"))
        self.Choose_MD_cB.setItemText(0, _translate("MainWindow", "PaDEL Descriptors"))
        self.Choose_MD_cB.setItemText(1, _translate("MainWindow", "RDKit Descriptors"))
        self.Draw_smile_pB.setText(_translate("MainWindow", "Draw"))
        self.Batch_calcu_tB.setText(_translate("MainWindow", "..."))
        self.Choose_MD_label.setText(_translate("MainWindow", "Choose Molecular Descriptors:"))
        self.Single_calcu_lineEdit.setText(_translate("MainWindow", "Input a SMILES"))
        self.Single_calcu_label.setText(_translate("MainWindow", "Single calculation:"))
        self.Batch_calcu_pB.setText(_translate("MainWindow", "Caculated (Batch)"))
        self.Batch_calcu_label.setText(_translate("MainWindow", "Batch calculation:"))
        self.Select_quantity_label.setText(_translate("MainWindow", "Select the quantity to be calculated: "))
        self.Select_quantity_cB.setItemText(0, _translate("MainWindow", "Just for Model"))
        self.Select_quantity_cB.setItemText(1, _translate("MainWindow", "All"))
        self.action1.setText(_translate("MainWindow", "Open file"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave_as.setText(_translate("MainWindow", "Save as"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("PFAS-Soil-Kd V1.0")
        QApplication.setStyle(QStyleFactory.create('fusion'))
    def Data_input(self):
        self.Data_input_Edit.setText('')
        self.tableWidget.clear()
        self.predict_text.setText('')
        Filename,Filetype = QFileDialog.getOpenFileName(self, "Select Data Input File", "./", "Files (*.xlsx *.csv)")
        if len(Filename) != 0:
            self.Data_input_Edit.setText(Filename)
        else:
            QMessageBox.about(self, 'Attention', 'Please select the path of file!')
    def prediction(self):
        Filename = self.Data_input_Edit.text()
        if len(Filename) != 0 and exists(Filename):
            if Filename.endswith('.xlsx'):
                data = read_excel(Filename)
            if Filename.endswith('.csv'):
                data = read_csv(Filename, encoding='gbk')
            Data = array(data.loc[:,'SOC':'logCe'])
            Model_choose = self.Model_choose_Box.currentText()
            if Model_choose == 'LightGBM trained by RDKit Dataset':
                Model = load('Model\\RDKit_LGBM.pkl')
                X_minmax = load('Model\\RDKit_X_minmax.pkl')
                Y_minmax = load('Model\\RDKit_Y_minmax.pkl')
                if Data.shape[0] == 0 or Data.shape[1] != 15:
                    QMessageBox.about(self, 'Attention', 'Data exception!Please reselect the file or adjust the selected model.')
                else:
                    input_data = Data
                    X_input = X_minmax.transform(input_data)
                    pred = Model.predict(X_input)
                    pred_Y = Y_minmax.inverse_transform(pred.reshape(-1, 1))
                    pred_Y = array(pred_Y)
                    result = hstack((input_data, pred_Y))
                    results = DataFrame(result,
                                            columns=['SOC', 'CEC', 'Sand','Silt', 'Clay',
                                                     'PEOE_VSA4', 'Chi3v', 'FpDensityMorgan3', 'NumValenceElectrons', 'MinAbsEStateIndex', 'MinPartialCharge',
                                                     'pH', 'Rws', 'PCS', 'logCe', 'pred_logKd'])
                    self.predict_text.setText('Model predict successfully! The number of predictive data is ' + str(len(input_data[:, 0])) + '.')
                    self.tableWidget.setRowCount(results.shape[0])
                    self.tableWidget.setColumnCount(results.shape[1])
                    for row in range(self.tableWidget.rowCount()):
                        for col in range(self.tableWidget.columnCount()):
                            item = QTableWidgetItem(str(results.iloc[row, col]))
                            self.tableWidget.setItem(row, col, item)
                    self.tableWidget.setHorizontalHeaderLabels(results.columns)
            if Model_choose == 'LightGBM trained by PaDEL Dataset':
                Model = load('Model\\PaDEL_LGBM.pkl')
                X_minmax = load('Model\\PaDEL_X_minmax.pkl')
                Y_minmax = load('Model\\PaDEL_Y_minmax.pkl')
                if Data.shape[0] == 0 or Data.shape[1] != 35:
                    QMessageBox.about(self, 'Attention',
                                      'Data exception!Please reselect the file or adjust the selected model.')
                else:
                    input_data = Data
                    X_input = X_minmax.transform(input_data)
                    pred = Model.predict(X_input)
                    pred_Y = Y_minmax.inverse_transform(pred.reshape(-1, 1))
                    pred_Y = array(pred_Y)
                    result = hstack((input_data, pred_Y))
                    results = DataFrame(result,
                                           columns=['SOC', 'CEC', 'Sand', 'Slit', 'Clay',
                                                    'ATS3s', 'SpMin8_Bhi', 'BIC3', 'SpMax3_Bhm', 'GATS2c', 'GATS6c', 'GATS3c', 'nRotBt',
                                                    'SpMax1_Bhi', 'AATS8p', 'MATS4s', 'MLFER_BO', 'AATSC5s', 'GATS1c', 'MATS1s', 'VE3_Dzp',
                                                    'ETA_BetaP', 'ATSC7p', 'minsOm', 'ATSC5v', 'MATS8i', 'GATS7p', 'ATSC7i', 'MATS8c', 'MIC3', 'ATSC7s',
                                                    'pH', 'Rws', 'PCS', 'logCe', 'pred_logKd'])
                    self.predict_text.setText('Model predict successfully! The number of predictive data is ' + str(len(input_data[:,0])) + '.')
                    self.tableWidget.setRowCount(results.shape[0])
                    self.tableWidget.setColumnCount(results.shape[1])
                    for row in range(self.tableWidget.rowCount()):
                        for col in range(self.tableWidget.columnCount()):
                            item = QTableWidgetItem(str(results.iloc[row, col]))
                            self.tableWidget.setItem(row, col, item)
                    self.tableWidget.setHorizontalHeaderLabels(results.columns)
        else:
            QMessageBox.about(self, 'Attention', 'Please select the path of file or check if the path is correct!')
    def Data_output(self):
        self.Data_output_Edit.setText('')
        self.Save_text.setText('')
        Filename, Filetype = QFileDialog.getSaveFileName(self, "Creat Data Output File", "./", "xlsx (*.xlsx);; csv(*.csv);; txt(*.txt)")
        if len(Filename) != 0:
            self.Data_output_Edit.setText(Filename)
        else:
            QMessageBox.about(self, 'Attention', 'Please select the path of file！')
    def Save(self):
        Filename = self.Data_output_Edit.text()
        if len(Filename) != 0:
            feature_columns = []
            for f in range(self.tableWidget.columnCount()):
                feature_columns.append(self.tableWidget.horizontalHeaderItem(f).text())
            result = DataFrame(columns=feature_columns)
            for row in range(self.tableWidget.rowCount()):
                for col in range(self.tableWidget.columnCount()):
                    item = self.tableWidget.item(row, col)
                    result.at[row, feature_columns[col]] = item.text() if item is not None else ""
            Notice = 'Result saved successfully!'
            if Filename.endswith('.xlsx'):
                result.to_excel(Filename, index=False)
                self.Save_text.setText(Notice)
            if Filename.endswith('.csv'):
                result.to_csv(Filename, index=False)
                self.Save_text.setText(Notice)
            if Filename.endswith('.txt'):
                result.to_csv(Filename, sep =',', index=False)
                self.Save_text.setText(Notice)
        else:
            self.Save_text.setText('')
            QMessageBox.about(self, 'Attention', 'Please select the path of file or check if the path is correct!')
    def clear_all(self):
        self.Data_input_Edit.setText('')
        self.Data_output_Edit.setText('')
        self.Save_text.setText('')
        self.tableWidget.clear()
        self.predict_text.setText('')
    def ui_trans(self,i):
        self.stackedWidget.setCurrentIndex(i)
    def Open_trainset_file(self):
        self.Open_trainset_file_lineEdit.setText('')
        Filename,Filetype = QFileDialog.getOpenFileName(self, "Select Data Input File", "./", "Files (*.xlsx *.csv)")
        if len(Filename) != 0:
            self.Open_trainset_file_lineEdit.setText(Filename)
        else:
            QMessageBox.about(self, 'Attention', 'Please select the path of file！')
    def Bayesian_optimization(self):
        random_seed = 205
        path = self.Open_trainset_file_lineEdit.text()
        if len(path) != 0 and exists(path):
            if path.endswith('.xlsx'):
                data = read_excel(path)
            if path.endswith('.csv'):
                data = read_csv(path, encoding='gbk')
            data = data.loc[:, 'SOC':'logKd']
            self.Bayesian_textBrowser.append('Data reading completed!')
            Data = array(data)
            n = len(Data[0, :])
            Data = shuffle(Data, random_state=random_seed)
            gkf = GroupKFold(n_splits=9)
            length = len(Data)
            group = zeros((length, 1))
            N = 0
            Ni = n - 5
            for i in range(length - 1):
                if group[i, 0] == 0:
                    N = N + 1
                    data1 = Data[i, 0:Ni]
                    group[i, 0] = N
                    for j in range(i + 1, length):
                        data2 = Data[j, 0:Ni]
                        if all(data1 == data2):
                            group[j, 0] = N
            nn = n - 1
            self.Bayesian_textBrowser.append('Start Bayesian optimization:')
            self.Bayesian_textBrowser.ensureCursorVisible()
            def BayesianSearch(clf, params):
                num_iter = int(self.Iterations_lineEdit.text())-1
                init_points = 0
                bayes = BayesianOptimization(clf, params, random_state=random_seed)
                bayes.maximize(init_points=init_points, n_iter=num_iter)
                max_params = bayes.max
                self.Bayesian_textBrowser.append(str(max_params))
                return max_params

            def GBM_evaluate(learning_rate, n_estimators,  max_depth, num_leaves, reg_alpha, reg_lambda, drop_rate, max_drop):
                if self.stop:
                    self.Bayesian_textBrowser.append('Bayesian optimization process interrupt!')
                    sys.exit(1)
                self.num = self.num + 1
                if int(num_leaves) < 2 ** int(max_depth):
                    param = {'objective': 'regression_l1',
                             'metric': 'mae',
                             'boosting': 'dart',
                             'random_state': 205}
                    param['n_estimators'] = int(n_estimators)
                    param['num_leaves'] = int(num_leaves),
                    param['max_depth'] = int(max_depth),
                    param['max_drop'] = int(max_drop),
                    param['reg_lambda'] = round(float(reg_lambda), 1),
                    param['reg_alpha'] = round(float(reg_alpha), 1),
                    param['learning_rate'] = round(float(learning_rate), 4),
                    param['drop_rate'] = round(float(drop_rate), 4),
                    result_r2 = []
                    result_mae = []
                    result_rmse = []
                    for train, test in gkf.split(Data, groups=group):
                        train_set = Data[train]
                        test_set = Data[test]
                        X_tr = train_set[:, 0:nn]
                        Y_tr = train_set[:, nn].reshape(-1, 1)
                        X_te = test_set[:, 0:nn]
                        Y_te = test_set[:, nn].reshape(-1, 1)
                        X_minmax = MinMaxScaler(feature_range=(-1, 1))
                        X_minmax.fit(X_tr)
                        X_train = X_minmax.transform(X_tr)
                        X_test = X_minmax.transform(X_te)
                        Y_minmax = MinMaxScaler(feature_range=(-1, 1))
                        Y_minmax.fit(Y_tr)
                        Y_train = Y_minmax.transform(Y_tr)
                        Y_test = Y_minmax.transform(Y_te)
                        modle = LGBMRegressor(**param,verbosity=-1)
                        modle.fit(X_train, Y_train.ravel())
                        pred_test = modle.predict(X_test)
                        pre_te = Y_minmax.inverse_transform(pred_test.reshape(-1, 1))
                        Y_te = Y_minmax.inverse_transform(Y_test.reshape(-1, 1))
                        r2_test = r2_score(Y_te, pre_te)
                        mae = mean_absolute_error(Y_te, pre_te)
                        rmse = sqrt(mean_squared_error(Y_te, pre_te))
                        result_r2.append(r2_test)
                        result_mae.append(mae)
                        result_rmse.append(rmse)
                    result_r2 = array(result_r2)
                    result_mae = array(result_mae)
                    result_rmse = array(result_rmse)
                    self.Bayesian_textBrowser.append(str(int(self.num))+'Iterations'+' '+'R'+ "\u00b2"+':' + str(result_r2.mean())
                                                     +' '+'RMSE'+':' + str(result_rmse.mean()) +' '+'MAE'+ ':' + str(result_mae.mean()))
                    return 1 / result_mae.mean()
                else:
                    self.Bayesian_textBrowser.append(
                        str(int(self.num)) + 'Iterations' + ' ' + 'num_leaves > 2^max_depth, unreasonable parameters')
                    return 0
            if __name__ == '__main__':
                adj_params = self.Optimization_range_TextEdit.toPlainText()
                adj_params = eval(adj_params)
                self.num = 0
                BayesianSearch(GBM_evaluate, adj_params)
    def Start_Bayesian(self):
        self.Bayesian_textBrowser.setText('')
        self.stop = False
        t = Thread(target=self.Bayesian_optimization, name='t')
        t.start()
    def Stop_Bayesian(self):
        self.stop = True
    def Start_retrain(self):
        random_seed = 205
        def model_train_result(self, data):
            random_seed = 205
            Data = array(data)
            n = len(Data[0, :])
            Data = shuffle(Data, random_state=random_seed)
            gkf = GroupKFold(n_splits=9)
            length = len(Data)
            group = zeros((length, 1))
            N = 0
            Ni = n - 5
            for i in range(length - 1):
                if group[i, 0] == 0:
                    N = N + 1
                    data1 = Data[i, 0:Ni]
                    group[i, 0] = N
                    for j in range(i + 1, length):
                        data2 = Data[j, 0:Ni]
                        if all(data1 == data2):
                            group[j, 0] = N
            nn = n - 1
            result_r2 = []
            result_rmse = []
            result_mae = []
            for train, test in gkf.split(Data, groups=group):
                train_set = Data[train]
                test_set = Data[test]
                X_tr = train_set[:, 0:nn]
                Y_tr = train_set[:, nn].reshape(-1, 1)
                X_te = test_set[:, 0:nn]
                Y_te = test_set[:, nn].reshape(-1, 1)
                X_minmax = MinMaxScaler(feature_range=(-1, 1))
                X_minmax.fit(X_tr)
                X_train = X_minmax.transform(X_tr)
                X_test = X_minmax.transform(X_te)
                Y_minmax = MinMaxScaler(feature_range=(-1, 1))
                Y_minmax.fit(Y_tr)
                Y_train = Y_minmax.transform(Y_tr)
                Y_test = Y_minmax.transform(Y_te)
                model.fit(X_train, Y_train.ravel())
                pred_test = model.predict(X_test)
                pre_te = Y_minmax.inverse_transform(pred_test.reshape(-1, 1))
                Y_te = Y_minmax.inverse_transform(Y_test.reshape(-1, 1))
                r2_test = r2_score(Y_te, pre_te)
                rmse = sqrt(mean_squared_error(Y_te, pre_te))
                mae = mean_absolute_error(Y_te, pre_te)
                result_r2.append(r2_test)
                result_rmse.append(rmse)
                result_mae.append(mae)
            result_r2 = array(result_r2).mean()
            result_rmse = array(result_rmse).mean()
            result_mae = array(result_mae).mean()
            self.R2_lineEdit.setText(str(around(result_r2,4)))
            self.RMSE_lineEdit.setText(str(around(result_rmse,4)))
            self.MAE_lineEdit.setText(str(around(result_mae,4)))
        if self.Learning_rate_lineEdit.text() == '' and self.Max_depth_lineEdit.text() == '' and self.N_estimat_lineEdit.text() == '' and self.Num_leaves_lineEdit.text() == '':
            QMessageBox.about(self, 'Attention', 'Parameters cannot be empty.')
        elif self.Reg_alpha_lineEdit.text() == '' and self.Reg_lambda_lineEdit.text() == '' and self.Max_drop_lineEdit.text() == '' and self.Drop_rate_lineEdit.text() == '':
            QMessageBox.about(self, 'Attention','Parameters cannot be empty.')
        else:
            Learning_rate = float(self.Learning_rate_lineEdit.text())
            Max_depth = int(self.Max_depth_lineEdit.text())
            N_estimators = int(self.N_estimat_lineEdit.text())
            Num_leaves = int(self.Num_leaves_lineEdit.text())
            Reg_alpha = float(self.Reg_alpha_lineEdit.text())
            Reg_lambda = float(self.Reg_lambda_lineEdit.text())
            Drop_rate = float(self.Drop_rate_lineEdit.text())
            Max_drop = int(self.Max_drop_lineEdit.text())
            model = LGBMRegressor(objective = 'regression_l1',
                                  metric = 'mae',
                                  boosting = 'dart',
                                  learning_rate=Learning_rate,
                                  max_depth=int(Max_depth),
                                  n_estimators=int(N_estimators),
                                  num_leaves=int(Num_leaves),
                                  reg_alpha=Reg_alpha,
                                  reg_lambda=Reg_lambda,
                                  drop_rate = Drop_rate,
                                  max_drop = Max_drop,
                                  random_state=random_seed)
            path = self.Open_trainset_file_lineEdit.text()
            if len(path) != 0 and exists(path):
                if path.endswith('.xlsx'):
                    data = read_excel(path)
                if path.endswith('.csv'):
                    data = read_csv(path, encoding='gbk')
                data = data.loc[:, 'SOC':'logKd']
                train_x = array(data.loc[:,'SOC':'logCe'])
                train_y = array(data.loc[:,'logKd']).reshape(-1,1)
                retrain_model_choose = self.Model_choose_Box_2.currentText()
                if retrain_model_choose == 'LightGBM trained by RDKit Dataset':
                    if data.shape[0] == 0 or data.shape[1] != 16:
                        QMessageBox.about(self, 'Attention',
                                          'Data exception!Please reselect the file or adjust the selected model.')
                    else:
                        model_train_result(self, data)
                        if self.checkBox.isChecked():
                            respond = QMessageBox.question(self,'Attention',
                                                 'Saving the current model will overwrite the previous model. Do you want to continue?', QMessageBox.Yes | QMessageBox.No)
                            if respond == 65536:
                                QMessageBox.about(self,'Attention','Model save canceled!')
                            else:
                                X_minmax = MinMaxScaler(feature_range=(-1, 1))
                                X_minmax.fit(train_x)
                                train_x = X_minmax.transform(train_x)
                                Y_minmax = MinMaxScaler(feature_range=(-1, 1))
                                Y_minmax.fit(train_y)
                                train_y = Y_minmax.transform(train_y)
                                model.fit(train_x, train_y.ravel())
                                dump(X_minmax, 'Model/RDKit_X_minmax.pkl')
                                dump(Y_minmax, 'Model/RDKit_Y_minmax.pkl')
                                dump(model, 'Model/RDKit_LGBM.pkl')
                                QMessageBox.about(self, 'Save Model',
                                                  'Current model has been saved, and the original model has been replaced!')
                        else:
                            QMessageBox.about(self, 'Retain Model',
                                              'Model training completed!')
                elif retrain_model_choose == 'LightGBM trained by PaDEL Dataset':
                    if data.shape[0] == 0 or data.shape[1] != 36:
                        QMessageBox.about(self, 'Attention',
                                          'Data exception!Please reselect the file or adjust the selected model.')
                    else:
                        model_train_result(self, data)
                        if self.checkBox.isChecked():
                            respond = QMessageBox.question(self,'Attention',
                                                 'Saving the current model will overwrite the previous model. Do you want to continue?', QMessageBox.Yes | QMessageBox.No)
                            if respond == 65536:
                                QMessageBox.about(self,'Attention','Model save canceled!')
                            else:
                                X_minmax = MinMaxScaler(feature_range=(-1, 1))
                                X_minmax.fit(train_x)
                                train_x = X_minmax.transform(train_x)
                                Y_minmax = MinMaxScaler(feature_range=(-1, 1))
                                Y_minmax.fit(train_y)
                                train_y = Y_minmax.transform(train_y)
                                model.fit(train_x, train_y.ravel())
                                dump(X_minmax, 'Model/PaDEL_X_minmax.pkl')
                                dump(Y_minmax, 'Model/PaDEL_Y_minmax.pkl')
                                dump(model, 'Model/PaDEL_LGBM.pkl')
                                QMessageBox.about(self, 'Save Model',
                                                  'Current model has been saved, and the original model has been replaced!')
                        else:
                            QMessageBox.about(self, 'Retain Model',
                                              'Model training completed!')
    def smile_draw(self):
        Single_SMILES = self.Single_calcu_lineEdit.text()
        if Single_SMILES !='':
            try:
                mol = MolFromSmiles(Single_SMILES)
                Draw.ShowMol(mol, size=(500, 500), kekulize=False)
            except Exception as e:
                message = f"An exception of type {type(e).__name__} occurred.\n{e}"
                QMessageBox.critical(None, "Error", message)
                return False
    def Single_calculate(self):
        self.MD_result_table.clear()
        Choose_MD = self.Choose_MD_cB.currentText()
        Select_quantity = self.Select_quantity_cB.currentText()
        Single_SMILES = self.Single_calcu_lineEdit.text()
        def table_setting(self,result):
            self.MD_result_table.setRowCount(result.shape[0])
            self.MD_result_table.setColumnCount(result.shape[1])
            for row in range(self.MD_result_table.rowCount()):
                for col in range(self.MD_result_table.columnCount()):
                    item = QTableWidgetItem(str(result.iloc[row, col]))
                    self.MD_result_table.setItem(row, col, item)
            self.MD_result_table.setHorizontalHeaderLabels(result.columns)
        self.MD_result_progressBar.setRange(0, 1)
        self.MD_result_progressBar.setValue(0)
        if Single_SMILES != '':
            try:
                if Choose_MD == 'RDKit Descriptors':
                    descriptor_names = [x[0] for x in Descriptors._descList]
                    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                    mol = MolFromSmiles(Single_SMILES)
                    RDKit_result = calculator.CalcDescriptors(mol)
                    RDKit_result = array(RDKit_result).reshape(1, -1)
                    RDKit_result = DataFrame(RDKit_result, columns=descriptor_names)
                    if RDKit_result.iloc[0,0] == -666.0:
                        QMessageBox.critical(None, "Error", "Ensure input structure is correct.")
                    else:
                        self.MD_result_progressBar.setValue(1)
                        if Select_quantity == 'Just for Model':
                            descriptors_model = ['PEOE_VSA4', 'Chi3v', 'FpDensityMorgan3', 'NumValenceElectrons', 'MinAbsEStateIndex', 'MinPartialCharge']
                            RDKit_result_model = RDKit_result.loc[:, descriptors_model]
                            table_setting(self, RDKit_result_model)
                            QMessageBox.about(self, 'Attention', 'RDKit descriptors just for model have been calculated!')
                        elif Select_quantity == 'All':
                            table_setting(self, RDKit_result)
                            QMessageBox.about(self, 'Attention', 'All RDKit descriptors have been calculated!')
                if Choose_MD == 'PaDEL Descriptors':
                    PaDEL_result = from_smiles(Single_SMILES)
                    PaDEL_result = DataFrame([PaDEL_result])
                    self.MD_result_progressBar.setValue(1)
                    if Select_quantity == 'Just for Model':
                        descriptors_model = ['ATS3s',	'SpMin8_Bhi',	'BIC3',	'SpMax3_Bhm',	'GATS2c',	'GATS6c',	'GATS3c',	'nRotBt',
                                             'SpMax1_Bhi',	'AATS8p',	'MATS4s',	'MLFER_BO',	'AATSC5s',	'GATS1c',	'MATS1s',	'VE3_Dzp',
                                             'ETA_BetaP',	'ATSC7p',	'minsOm',	'ATSC5v',	'MATS8i',	'GATS7p',	'ATSC7i',	'MATS8c',	'MIC3',	'ATSC7s']
                        PaDEL_result_model = PaDEL_result.loc[:, descriptors_model]
                        table_setting(self, PaDEL_result_model)
                        QMessageBox.about(self, 'Attention', 'PaDEL descriptors just for model have been calculated!')
                    elif Select_quantity == 'All':
                        table_setting(self, PaDEL_result)
                        QMessageBox.about(self, 'Attention', 'All PaDEL descriptors have been calculated!')
            except Exception as e:
                message = f"An exception of type {type(e).__name__} occurred.\n{e}"
                QMessageBox.critical(self, "Error", message)
                return False
    def Open_smiles_file(self):
        self.Batch_calcu_lineEdit.setText('')
        Filename,Filetype = QFileDialog.getOpenFileName(self, "Select Data Input File", "./", "Files (*.xlsx *.csv)")
        if len(Filename) != 0:
            self.Batch_calcu_lineEdit.setText(Filename)
        else:
            QMessageBox.about(self, 'Attention', 'Please select the path of file！')
    def Batch_calculate(self):
        self.MD_result_table.clear()
        self.MD_result_progressBar.setValue(0)
        Choose_MD = self.Choose_MD_cB.currentText()
        Select_quantity = self.Select_quantity_cB.currentText()
        Batch_SMILES = self.Batch_calcu_lineEdit.text()
        def table_setting(self,result):
            self.MD_result_table.setRowCount(result.shape[0])
            self.MD_result_table.setColumnCount(result.shape[1])
            for row in range(self.MD_result_table.rowCount()):
                for col in range(self.MD_result_table.columnCount()):
                    item = QTableWidgetItem(str(result.iloc[row, col]))
                    self.MD_result_table.setItem(row, col, item)
            self.MD_result_table.setHorizontalHeaderLabels(result.columns)
        if len(Batch_SMILES) != 0 and exists(Batch_SMILES):
            if Batch_SMILES.endswith('.xlsx'):
                smiles_data = read_excel(Batch_SMILES)
            if Batch_SMILES.endswith('.csv'):
                smiles_data = read_csv(Batch_SMILES, encoding='gbk')
            try:
                smiles = smiles_data.loc[:,'SMILES']
            except Exception as e:
                message = f"An exception of type {type(e).__name__} occurred.\nPlease check if the table column name is SMILES!"
                QMessageBox.critical(self, "Error", message)
                return False
            else:
                self.MD_result_progressBar.setRange(0,len(smiles))
                self.MD_result_progressBar.setValue(0)
                if len(smiles) != 0:
                    try:
                        for i in range(len(smiles)):
                            mol = MolFromSmiles(smiles.iloc[i])
                    except Exception as e:
                        message = f"An exception of type {type(e).__name__} occurred.\nPlease check if the SMILES in the file are correct!"
                        QMessageBox.critical(self, "Error", message)
                        return False
                    else:
                        results = []
                        if Choose_MD == 'RDKit Descriptors':
                            descriptor_names = [x[0] for x in Descriptors._descList]
                            calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                            for i in range(len(smiles)):
                                mol = MolFromSmiles(smiles.loc[i])
                                result = calculator.CalcDescriptors(mol)
                                results.append(result)
                                self.MD_result_progressBar.setValue(i+1)
                            RDKit_result = DataFrame(results, columns=descriptor_names)
                            if Select_quantity == 'Just for Model':
                                descriptors_model = ['PEOE_VSA4', 'Chi3v', 'FpDensityMorgan3', 'NumValenceElectrons', 'MinAbsEStateIndex', 'MinPartialCharge']
                                RDKit_result_model = RDKit_result.loc[:, descriptors_model]
                                table_setting(self, RDKit_result_model)
                                QMessageBox.about(self, 'Attention', 'RDKit descriptors just for model have been calculated!')
                            elif Select_quantity == 'All':
                                table_setting(self, RDKit_result)
                                QMessageBox.about(self, 'Attention', 'All RDKit descriptors have been calculated!')
                        if Choose_MD == 'PaDEL Descriptors':
                            PaDEL_result = DataFrame()
                            for i in range(len(smiles)):
                                result = from_smiles(smiles.iloc[i],threads = 2)
                                result = DataFrame([result])
                                if i == 0:
                                    PaDEL_result = result
                                else:
                                    PaDEL_result = concat([PaDEL_result,result])
                                self.MD_result_progressBar.setValue(i + 1)
                            if Select_quantity == 'Just for Model':
                                descriptors_model = ['ATS3s',	'SpMin8_Bhi',	'BIC3',	'SpMax3_Bhm',	'GATS2c',	'GATS6c',	'GATS3c',	'nRotBt',
                                             'SpMax1_Bhi',	'AATS8p',	'MATS4s',	'MLFER_BO',	'AATSC5s',	'GATS1c',	'MATS1s',	'VE3_Dzp',
                                             'ETA_BetaP',	'ATSC7p',	'minsOm',	'ATSC5v',	'MATS8i',	'GATS7p',	'ATSC7i',	'MATS8c',	'MIC3',	'ATSC7s']
                                PaDEL_result_model = PaDEL_result.loc[:,descriptors_model]
                                table_setting(self, PaDEL_result_model)
                                QMessageBox.about(self, 'Attention', 'PaDEL descriptors just for model have been calculated!')
                            elif Select_quantity == 'All':
                                table_setting(self, PaDEL_result)
                                QMessageBox.about(self, 'Attention', 'All PaDEL descriptors have been calculated!')
    def Export_MD_data(self):
        item = self.MD_result_table.item(0, 0)
        if item is None:
            QMessageBox.about(self, 'Attention', 'Empty table cannot export data!')
        else:
            Filename, Filetype = QFileDialog.getSaveFileName(self, "Export Data", "./",
                                                         "xlsx (*.xlsx);; csv(*.csv);; txt(*.txt)")
            if len(Filename) == 0:
                QMessageBox.about(self, 'Attention', 'Please select the path of file!')
            else:
                feature_columns = []
                for f in range(self.MD_result_table.columnCount()):
                    feature_columns.append(self.MD_result_table.horizontalHeaderItem(f).text())
                MD_result = DataFrame(columns=feature_columns)
                for row in range(self.MD_result_table.rowCount()):
                    for col in range(self.MD_result_table.columnCount()):
                        item = self.MD_result_table.item(row, col)
                        MD_result.at[row, feature_columns[col]] = item.text() if item is not None else ""
                if Filename.endswith('.xlsx'):
                    MD_result.to_excel(Filename, index=False)
                if Filename.endswith('.csv'):
                    MD_result.to_csv(Filename, index=False)
                if Filename.endswith('.txt'):
                    MD_result.to_csv(Filename, sep=',', index=False)
                QMessageBox.about(self, 'Attention', 'Export data successfully!')
    def click_button(self):
        self.Start_Button.clicked.connect(self.prediction)
        self.Data_input_Button.clicked.connect(self.Data_input)
        self.Data_output_Button.clicked.connect(self.Data_output)
        self.Save_Button.clicked.connect(self.Save)
        self.Clear_Button.clicked.connect(self.clear_all)
        self.Pred_tB.clicked.connect(lambda : self.ui_trans(0))
        self.Retrain_tB.clicked.connect(lambda : self.ui_trans(1))
        self.Descrip_tB.clicked.connect(lambda : self.ui_trans(2))
        self.Open_retrainset_file_pB.clicked.connect(self.Open_trainset_file)
        self.Bayesian_pB.clicked.connect(self.Start_Bayesian)
        self.Stop_pB.clicked.connect(self.Stop_Bayesian)
        self.Start_retraining_pB.clicked.connect(self.Start_retrain)
        self.Draw_smile_pB.clicked.connect(self.smile_draw)
        self.Single_calcu_pB.clicked.connect(self.Single_calculate)
        self.Batch_calcu_tB.clicked.connect(self.Open_smiles_file)
        self.Batch_calcu_pB.clicked.connect(self.Batch_calculate)
        self.Export_data_pB.clicked.connect(self.Export_MD_data)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.setWindowIcon(QIcon('/icon/soil_pollution.ico'))
    myWin.show()
    myWin.click_button()
    sys.exit(app.exec_())

