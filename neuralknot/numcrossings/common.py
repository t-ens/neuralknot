import numpy as np

import matplotlib.pyplot as plt

#AMD tensorflow docker image doesn't export image_dataset_from_directoy like usual
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.data import AUTOTUNE

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Sequential

from neuralknot.common import KnotModel

class NumCrossings(KnotModel):
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

    def visualize_data(self, num=9):
        plt.figure()
        for images,labels in self.train_ds.take(1):
            for i in range(num):
                ax = plt.subplot(3,3,i+1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i].numpy().astype("uint8")])
                plt.axis('off')
        plt.show()
