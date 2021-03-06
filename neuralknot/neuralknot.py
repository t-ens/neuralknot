import sys

_QT = True
try:
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError:
    _QT = False

_QT_MODERN = True
try:
    from qtmodern import styles, windows
except ModuleNotFoundError:
    _QT_MODERN = False


from tensorflow.config.threading import set_intra_op_parallelism_threads
from tensorflow.config.threading import set_inter_op_parallelism_threads

if _QT:
    from neuralknot.qtgui.mainwindow import MainWindow

from neuralknot.numcrossings.blockconv import BlockModel
from neuralknot.numcrossings.fullconv import FullConv
from neuralknot.gaussencoder.simpleGRU import SimpleGRU
from neuralknot.gaussencoder.doublebiGRU import DoubleBiGRU

set_intra_op_parallelism_threads(8)
set_inter_op_parallelism_threads(8) 

def main_cli():
    """Very simple interactive loop to load data and models, visualize either
    and train models
    """
    
    current_model = BlockModel()

    while True:
        print(f'Selected Model: {current_model._net_name}:{current_model._model_name}') 
        print('Options:')
        print('  1) Select model')
        print('  2) Plot model history')
        print('  3) Visualize data')
        print('  4) Plot model graph')
        print('  5) Evaluate model')
        print('  6) Train model')
        print('  7) Predict with model')
        print('  8) Exit')
        choice = input('Enter number: ')

        if choice == '1':
            print('Available networks:')
            print('  1) blockconv (numcrossings)')
            print('  2) fullconv (numcrossings)')   
            print('  3) simpleGRU (gaussencoder)')
            print('  4) doubleBiGRU (gaussencoder)')
            selection = input('Choice: ')

            if selection == '1':
                current_model = BlockModel()
            elif selection == '2':
                current_model = FullConv()
            elif selection == '3':
                current_model = SimpleGRU()
            elif selection == '4':
                current_model = DoubleBiGRU()
            
        elif choice == '2':
            current_model.plot_history()

        elif choice == '3':
            current_model.visualize_data()

        elif choice == '4':
            current_model.plot_model()

        elif choice == '5':
            current_model.evaluate_model()

        elif choice == '6':
            epochs = input('How many epochs? ')
            try:
                epochs = int(epochs)
            except ValueError:
                print('Number of epochs must be a positive integer')
            if epochs>0:
                current_model.train_model(epochs)
            
        elif choice == '7':
            num_predictions = input('How many predictions (max 32): ')
            valid_input = True
            try:
                num_predictions = int(num_predictions)
            except ValueError:
                valid_input = False

            if  valid_input and 0 <= num_predictions <= 32:
                current_model.predict(num_predictions)
            else:
                print('Number of predictions must be a positive integer')

        elif choice == '9' or 'q':
            return 0

def main_qt():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.menuBar()

    if _QT_MODERN:
        styles.dark(app)
        mwin = windows.ModernWindow(win)
        mwin.show()
    else:
        win.show()

    return app.exec()

def main():
    if _QT:
        return main_qt()
    else:
        return main_cli()
