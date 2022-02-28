import os
import glob
import re
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as img

import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import plot_model as keras_plot_model
from tensorflow.ragged import stack
from tensorflow.data import Dataset
from tensorflow.data import AUTOTUNE
from tensorflow.strings import unicode_split

from tensorflow.keras.layers import StringLookup
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.train import latest_checkpoint

class GaussEncoder:
    """
        This class contains methods and objects common to all models for gauss
        encoding. Do not instantiate this class directly, you should always
        instantiate the model class.
    """
    def __init__(self):
        self._net_name = 'gaussencoder'
        
        self._base_dir = '/'.join(['neuralknot', self._net_name])
        self._data_dir = '/'.join([self._base_dir, 'dataset'])
            
        # '*' is a padding character. '[' marks the beginning of a gauss code
        # and ']' marks the end. Everything else is either an integer, a minus
        # sign or a comma

        #This will allow knots with arbitrarily many crossings after training
        #the model on some finite set of knots. The model will probably work
        #better if the dictionary consists of of integers themselves rather than
        #individual digits
        self._dictionary = ['@', '-',',', '[',']'] + [ str(i) for i in range(10)]
        self._to_num = StringLookup(vocabulary=self._dictionary, oov_token="")
        self._to_char = StringLookup(vocabulary=self._to_num.get_vocabulary(), oov_token="", invert=True)

        self.train_ds, self.val_ds = self.load_data()

    def _parse_gauss_string(self, w):
        match = re.search( '.*?\[(.*)\]', w)
        code = match.groups(0)[0].replace(' ', '')
        return code

    def load_data(self, vsplit=0.2, image_size=(512,512), batch_size=32):
        """
            Load data from gaussencode/dataset. This is the wrong way of doing
            it since I am storing the same image file many time which will lead
            to massive datasets. This should be replaced with a data generator
            or something which will generate all the appropriate labellings from
            a single image instead
        """
        image_dir = '/'.join([self._data_dir, 'images'])

        with open('/'.join([self._data_dir, 'text_input.txt']), 'r') as fd:
            text_inputs = fd.readlines()
        with open('/'.join([self._data_dir, 'labels.txt']), 'r') as fd:
            labels = fd.readlines()

        #Remove spaces and newline characters
        text_inputs = [ line.replace(' ','') for line in text_inputs ]
        text_inputs = [ line.rstrip() for line in text_inputs ]
        #Encode using dictionary 
        text_inputs = [ self._to_num([ch for ch in line]) for line in text_inputs ]
        #Pad to length of longest input
        text_inputs = pad_sequences(text_inputs)
        self._label_length = text_inputs[0].shape[0]
        text_ds = Dataset.from_tensor_slices(text_inputs)

        labels= [ line.rstrip() for line in labels ]
        labels  = [ self._to_num(label) for label in labels ]
        label_ds = Dataset.from_tensor_slices(labels)

        fnames = glob.glob('/'.join([image_dir, '*']))
        num_images = len(fnames)
        images = image_dataset_from_directory(
                self._data_dir,
                labels=None,
                shuffle=False,
                image_size=image_size,
                batch_size=None,
                color_mode='grayscale')

        input_ds = Dataset.zip((images, text_ds))
        full_ds = Dataset.zip((input_ds, label_ds))

        val_num = round(num_images * vsplit)
        val_ds = (full_ds.take(val_num)
                    .padded_batch(32)
                    .prefetch(buffer_size=AUTOTUNE))
        train_ds = (full_ds.skip(val_num)
                    .padded_batch(32)
                    .prefetch(buffer_size=AUTOTUNE))

        return train_ds, val_ds

    def plot_model(self):
        if not os.path.isdir(self._model_dir):
            os.mkdir(self._model_dir)
        fname = '/'.join([self._model_dir, 'model_graph.png'])
        keras_plot_model(self.model, to_file=fname, show_shapes=True)
        plt.imshow(img.imread(fname))
        plt.axis('off')
        plt.show()

    def update_callbacks(self):
        sessions = self.get_training_sessions()
        try:
            next_session = max(sessions) + 1
        except ValueError:
            next_session = 0
        save_dir = '/'.join([self._model_dir, f'session_{next_session}/'])

        checkpoint = ModelCheckpoint(      
                filepath = save_dir + 'cp-{epoch:04d}.ckpt',
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

        if os.path.isfile('/'.join([save_dir,'checkpoint'])):
            latest = latest_checkpoint(save_dir)
            print(f'Loading weights from "{latest}"')
            self.model.load_weights(latest)
        else:
            print('Weights could not be loaded.')

    def plot_history(self):
        fnames = glob.glob( '/'.join([self._model_dir, 'session_?', 'history.csv']))
        fnames.sort(
            key= lambda w:  int(re.search('.*?session\_(.*)\/.*', w).groups(0)[0]))
    
        if len(fnames)>0:
            history = pd.read_csv(fnames[0]) 
            for fname in fnames[1:]:
                df = pd.read_csv(fname)
                history = pd.concat([history, df], axis=0, ignore_index=True)

        plt.figure()
        plt.plot(history['loss'], 'b')
        plt.plot(history['val_loss'], 'g')
        plt.legend(['loss', 'val_loss'])

        plt.figure()
        plt.plot(history['accuracy'], 'b')
        plt.plot(history['val_accuracy'], 'g')
        plt.legend(['acc', 'val_acc'])

        plt.show()

