import sys
from queue import Queue

from PySide6.QtCore import Qt, QThread, QObject, Signal, Slot
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QMenu, 
        QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
        QMessageBox, QTabWidget, QTreeWidget, QTreeWidgetItem)
from PySide6.QtGui import QAction, QDoubleValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from neuralknot.numcrossings.fullconv import FullConv
from neuralknot.numcrossings.blockconv import BlockModel
from neuralknot.gaussencoder.simpleGRU import SimpleGRU
from neuralknot.gaussencoder.doublebiGRU import DoubleBiGRU

class MatplotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, num=1):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor((0,0,0))
        self.fig.patch.set_alpha(0.2)


        if isinstance(num, int):
            self.axes = [ self.fig.add_subplot(num,1,i) for i in range(1,num+1) ]
        else:
            self.axes = [self.fig.add_subplot(num[0], num[1], i) for i in range(1, num[0]*num[1]+1) ]

        super(MatplotCanvas, self).__init__(self.fig)

class TrainModelThread(QThread):
    def __init__(self, epochs, model):
        QThread.__init__(self)

        self.epochs = epochs
        self.model = model

    def __del__(self):
        self.wait()

    def run(self):
        self.model.train_model(self.epochs)

class EvaluateModelThread(QThread):
    def __init__(self, model):
        QThread.__init__(self)
        self.model=model

    def __del__(self):
        self.wait()

    def run(self):
        self.model.evaluate_model()

class MainWindow(QMainWindow):
    """Main window for neuralknot QT GUI"""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.model = None

        self.setWindowTitle("Nerualknot")
        self.resize(600,400)

        self.centralWidget = QLabel('Load a model from the Model menu.')
        self.centralWidget.setAlignment(Qt.AlignHCenter | Qt. AlignVCenter)
        self.setCentralWidget(self.centralWidget)

        self.modelStatusLabel = QLabel("Current Model: No model loaded")
        self.statusBar().addPermanentWidget(self.modelStatusLabel)

        self.trainButton = QPushButton("Train")
        self.evaluateButton = QPushButton("Evaluate")

        self._createActions()
        self._createMenuBar()
        self._connectActions()

        self.out_box = None

    def _createActions(self):
        self.exitAction = QAction("&Exit", self)
        self.loadFullModel = QAction("&Full Convolution Model", self)
        self.loadBlockModel = QAction("&Block Convolution Model", self)
        self.loadSimpleGRUModel = QAction("&Simple GRU Model", self)
        self.loadDoubleBiGRUModel = QAction("&Double Bidirectional GRU Model", self)

    def _createMenuBar(self):
        menuBar = self.menuBar()

        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.exitAction)

        modelMenu = QMenu("&Model", self)
        menuBar.addMenu(modelMenu)
        loadMenu = modelMenu.addMenu("&Load model")
        loadMenu.addAction(self.loadFullModel)
        loadMenu.addAction(self.loadBlockModel)
        loadMenu.addAction(self.loadSimpleGRUModel)
        loadMenu.addAction(self.loadDoubleBiGRUModel)

    def _connectActions(self):
        self.exitAction.triggered.connect(self.close)
        self.loadFullModel.triggered.connect(lambda: self.loadModel(FullConv))
        self.loadBlockModel.triggered.connect(lambda: self.loadModel(BlockModel))
        self.loadSimpleGRUModel.triggered.connect(lambda: self.loadModel(SimpleGRU))
        self.loadDoubleBiGRUModel.triggered.connect(lambda: self.loadModel(DoubleBiGRU))
        self.trainButton.clicked.connect(
                lambda: self.train_model(int(self.epochInput.text()))
        )
        self.evaluateButton.clicked.connect(lambda: self.evaluate_model())
    
    def train_model(self, epochs):
        self.trainModelThread = TrainModelThread(epochs, self.model)
        self.trainModelThread.start()
        self.trainModelThread.finished.connect(self.createLoadedModelWindow)

    def updateEvalInfo(self):
        self.infoTree_eval.child(0).setText(0, f'Accuracy: {self.model.acc}')
        self.infoTree_eval.child(1).setText(0, f'Loss: {self.model.loss}')

    def evaluate_model(self):
        self.evaluateModelThread = EvaluateModelThread(self.model)
        self.evaluateModelThread.start()
        self.evaluateModelThread.finished.connect(self.updateEvalInfo)

    def loadModel(self, new_model):
        self.statusBar().showMessage('Loading new model')
        self.statusBar().repaint()
        self.model = new_model()
        self.statusBar().showMessage('Model loaded', 3000)
        self.modelStatusLabel.setText(f'Current Model: {self.model._model_name}')
        self.createLoadedModelWindow()


    def createLoadedModelWindow(self):
        ### The action box 
        self.epochInput = QLineEdit()
        self.epochInput.setMaximumWidth(100)
        self.epochInput.setValidator(QDoubleValidator(0, 1e6, 0))
        self.epochInput.setText('1')

        self.actionBoxTrainLayout = QHBoxLayout()
        self.actionBoxTrainLayout.addWidget(self.trainButton)
        self.actionBoxTrainLayout.addWidget(QLabel('for'))
        self.actionBoxTrainLayout.addWidget(self.epochInput)
        self.actionBoxTrainLayout.addWidget(QLabel('epoch(s)'))
        self.actionTrainBox = QWidget()
        self.actionTrainBox.setLayout(self.actionBoxTrainLayout)

        self.actionEvaluateBox = self.evaluateButton

        self.actionBoxLayout = QVBoxLayout()
        self.actionBoxLayout.addWidget(self.actionEvaluateBox)
        self.actionBoxLayout.addWidget(self.actionTrainBox)
        self.actionBox = QWidget()
        self.actionBox.setLayout(self.actionBoxLayout)

        self.infoTree = QTreeWidget()
        self.infoTree.setHeaderLabels(['Model Info'])
        self.infoTree_desc = QTreeWidgetItem(['Description'])
        self.infoTree_desc.addChild(QTreeWidgetItem([self.model.__desc__]))
        self.infoTree_eval = QTreeWidgetItem(['Current Model Evalutation'])
        self.infoTree_eval.addChild(QTreeWidgetItem(['Evaluate model First']))
        self.infoTree_eval.addChild(QTreeWidgetItem(['']))

        self.infoTree.addTopLevelItem(self.infoTree_desc)
        self.infoTree.addTopLevelItem(self.infoTree_eval)

        self.modelBoxLayout = QVBoxLayout()
        self.modelBoxLayout.addWidget(self.infoTree)
        self.modelBoxLayout.addStretch()
        self.modelBoxLayout.addWidget(self.actionBox)

        self.modelBox = QWidget()
        self.modelBox.setLayout(self.modelBoxLayout)
        self.modelBox.setMaximumWidth(300)

        self.historyCanvas = MatplotCanvas(self, width=5,height=4, num=2)
        self.historyCanvas.fig.suptitle('Model History')
        self.model.plot_history(self.historyCanvas.axes)
        
        self.modelCanvas = MatplotCanvas(self, width=5, height=4, num=1)
        self.modelCanvas.fig.suptitle('Model Graph')
        self.model.plot_model(self.modelCanvas.axes)

        self.dataCanvas = MatplotCanvas(self, width=5,height=4, num=(3,3))
        self.dataCanvas.fig.suptitle('Sample Data Plots')
        self.model.visualize_data(self.dataCanvas.axes, num=9)

        self.tabbedWindows = QTabWidget()
        self.tabbedWindows.addTab(self.historyCanvas, 'Model History')
        self.tabbedWindows.addTab(self.modelCanvas, 'Model Graph')
        self.tabbedWindows.addTab(self.dataCanvas, 'Visualize Data')

        self.mainGrid = QHBoxLayout()
        self.mainGrid.addWidget(self.modelBox)
        self.mainGrid.addWidget(self.tabbedWindows)

        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainGrid)
        self.setCentralWidget(self.centralWidget)
