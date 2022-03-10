import os
import glob
import re

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as img

from tensorflow.keras.utils import plot_model as keras_plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.train import latest_checkpoint

class KnotModel:
    """
        This class contains methods and objects common to all models. This class
        should not be instantiated; instantiate the model class itself.
    """

    def get_training_sessions(self):
        sessions = glob.glob('/'.join([self._model_dir, 'session_*']))
        return [ int(session.split('_')[-1]) for session in sessions ]

    def load_weights(self, mode = 'latest'):
        if mode == 'latest':
            try:
                num = max(self.get_training_sessions())
            except ValueError:
                print('No previous training sessions found')
                return 
            save_dir = '/'.join([self._model_dir, f'session_{int(num)}'])

        if os.path.isfile('/'.join([save_dir, 'checkpoint'])):
            latest = latest_checkpoint(save_dir)
            print(f'Loading weights from "{latest}"')
            self.model.load_weights(latest)
        else:
            print('Weights could not be loaded.')

    def plot_history(self, axes):
        fnames = glob.glob( '/'.join([self._model_dir, 'session_?', 'history.csv']))
        fnames.sort(
            key= lambda w:  int(re.search('.*?session\_(.*)\/.*', w).groups(0)[0]))
    
        if len(fnames)>0:
            history = pd.read_csv(fnames[0]) 
            columns = history.columns[1:]
            num_cols = len(columns)
            for fname in fnames[1:]:
                df = pd.read_csv(fname)
                history = pd.concat([history, df], axis=0, ignore_index=True)

            for i in range(num_cols//2):
                axes[i].plot(history[columns[i]], 'b')
                axes[i].plot(history[columns[num_cols//2+i]], 'g')
                axes[i].legend([columns[i], columns[num_cols//2+i]])
        else: 
            print("No training history found")

    def plot_model(self, axes):
        fname = '/'.join([self._model_dir, 'model_graph.png'])
        keras_plot_model(self.model, to_file=fname, show_shapes=True)
        axes[0].imshow(img.imread(fname))
        axes[0].axes.get_xaxis().set_visible(False)
        axes[0].axes.get_yaxis().set_visible(False)

    def update_callbacks(self):
        sessions = self.get_training_sessions()
        try:
            next_session = max(sessions) + 1
        except ValueError:
            next_session = 0
        save_dir = '/'.join([self._model_dir, f'session_{next_session}'])
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        checkpoint = ModelCheckpoint(      
            filepath = '/'.join([save_dir, 'cp-{epoch:04d}.ckpt']),
            save_weights_only=True,      
            verbose = 1)
    
        history_logger = CSVLogger(    
            '/'.join([save_dir,'history.csv']),    
            separator = ',',    
            append=True)
    
        return [checkpoint, history_logger]

    def train_model(self, epochs):
        callbacks = self.update_callbacks()
        history= self.model.fit(
                self.train_ds,
                validation_data = self.val_ds,
                epochs = epochs,
                callbacks=callbacks)

    def evaluate_model(self):
        self.loss, self.acc = self.model.evaluate(self.val_ds, verbose=2)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * self.acc))
