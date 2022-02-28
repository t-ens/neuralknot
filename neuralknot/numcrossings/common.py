import os
import glob
import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img

#AMD tensorflow docker image doesn't export image_dataset_from_directoy like usual
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import plot_model as keras_plot_model

from tensorflow.data import AUTOTUNE

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Sequential

class NumCrossings:
    def __init__(self):
        self._net_name = 'numcrossings'

        self._base_dir = '/'.join(['neuralknot', self._net_name])
        self._data_dir = '/'.join([self._base_dir, 'dataset', 'images'])

        self.train_ds, self.val_ds = self.load_data(self._data_dir)
        self.class_names = self.train_ds.class_names

    def load_data(self, folder, vsplit=0.2, image_size=(512,512), batch_size=32):    
        
        train_ds = image_dataset_from_directory(    
                    folder,    
                    seed = 123,    
                    validation_split = vsplit,    
                    subset='training',    
                    image_size=image_size,    
                    batch_size=batch_size,    
                    color_mode='grayscale')    
        
        val_ds = image_dataset_from_directory(    
                    folder,    
                    seed = 123,    
                    validation_split = vsplit,    
                    subset='training',    
                    image_size=image_size,    
                    batch_size=batch_size,    
                    color_mode='grayscale')    
        
        return train_ds, val_ds 

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
            self.model.load_weights(save_dir + '/')
        else:
            print('Weights could not be loaded.')

    def update_callbacks(self):
        sessions = self.get_training_sessions()
        try:
            next_session = max(sessions) + 1
        except ValueError:
            next_session = 0
        save_dir = '/'.join([self.model_dir, f'session_{next_session}/'])

        checkpoint = ModelCheckpoint(      
            filepath = save_dir,
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
                self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE),
                validation_data = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE),
                epochs = epochs,
                callbacks = callbacks)

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

    def visualize_data(self, num=9):
        plt.figure()
        for images,labels in self.train_ds.take(1):
            for i in range(num):
                ax = plt.subplot(3,3,i+1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i].numpy().astype("uint8")])
                plt.axis('off')
        plt.show()
        
    def plot_model(self):
        fname = '/'.join([self._model_dir, 'model_graph.png'])
        keras_plot_model(self.model, to_file=fname, show_shapes=True)
        plt.imshow(img.imread(fname))
        plt.axis('off')
        plt.show()

    def evaluate_model(self):
        loss, acc = self.model.evaluate(self.val_ds, verbose=2)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    def predict(self, num):
        predict_model = Sequential([self.model, Softmax()])

        for images, labels in self.val_ds.take(1):    
            predictions = self.model.predict(images)    
            for i in range(num):    
        
                plt.figure()    
        
                plt.subplot(2,2,1)    
                plt.imshow(images[i].numpy().astype("uint8"))    
                plt.title(self.class_names[labels[i]])    
                plt.axis('off')    
         
                plt.subplot(2,2,2)    
                plt.bar(range(len(self.class_names)), predictions[i])    
         
                max_inds = np.argpartition(predictions[i], -3)[-3:]    
                max_ind = np.argmax(predictions[i])    
                for j in max_inds:    
                    if j == max_ind:    
                        colour = 'r'    
                    else:    
                        colour = 'g'    
                    plt.text(
                            j, 
                            predictions[i][j], 
                            self.class_names[j], 
                            c=colour, 
                            rotation='45', 
                            fontsize='x-small')    
                    
                plt.show()
