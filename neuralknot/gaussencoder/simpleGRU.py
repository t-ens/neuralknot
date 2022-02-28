import glob
import numpy as np

import matplotlib.pyplot as plt
import cv2

from neuralknot.numcrossings.blockconv import Blocks 

from tensorflow.keras import Input
from tensorflow.keras.layers import  Layer
from tensorflow.keras import Model

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Concatenate, Dense, GRU, Embedding, Add

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences

from neuralknot.gaussencoder.common import GaussEncoder

class SimpleGRU(GaussEncoder):
    """
        This model follows the approach described in the paper "Show and Tell: A
        nerual Image Caption Generator" to convert an image to a sequence. An
        element of the dataset is an image and a partial Gauss code. Each
        datapoint is labelled by the next character in the Gauss code.
        Convolution layers are perfored on the image, and the resulting vector
        initializes the hidden state of the GRU layer. The partial Gauss code is
        then fed through the GRU layers to predict the output character. 
    """
    def __init__(self):
        self._model_name = 'simplegru'

        super().__init__()
        self._model_dir = '/'.join([self._base_dir, f'{self._model_name}_data'])
        
        self.model = self._make_model()
        self.load_weights()

    def _make_model(self,):
        image_input = Input(shape=((512,512,1)))
        
        x = Rescaling(1./255)(image_input)
        
        #Break image into 2x2, 4x4, 16x16 and 32x32 blocks
        x1 = Blocks(2)(x)
        x2 = Blocks(4)(x)
        x3 = Blocks(16)(x)
        x4 = Blocks(32)(x)
        
        #Run convolution on each block
        x1 = Conv2D(1,2, activation="relu")(x1)
        x1 = Flatten()(x1)
        
        x2 = Conv2D(1,2, activation="relu")(x2)
        x2 = MaxPool2D((2,2))(x2)
        x2 = Flatten()(x2)
        
        x3 = Conv2D(32,2, activation="relu")(x3)
        x3 = MaxPool2D((2,2))(x3)
        x3 = Conv2D(32,2, activation="relu")(x3)
        x3 = MaxPool2D((2,2))(x3)
        x3 = Flatten()(x3)
                                          
            
        x4 = Conv2D(32,2, activation="relu")(x4)
        x4 = MaxPool2D((2,2))(x4) 
        x4 = Conv2D(32,2, activation="relu")(x4)
        x4 = MaxPool2D((2,2))(x4)
        x4 = Flatten()(x4)
        
        x = Concatenate(axis=1)([x3,x4])
        x = Dense(200, activation='relu')(x)
    
        text_input = Input(shape=(self._label_length,))
        y = Embedding(self._to_num.vocabulary_size(), 50, mask_zero=True)(text_input)
        y = GRU(200, dropout=0.2)(y, initial_state=x)

        x = Add()([x,y])
        output = Dense(self._to_num.vocabulary_size(), activation="softmax")(x)
        
        model = Model((image_input, text_input), output, name="knot2gauss")
        
        model.compile(
                optimizer='adam',
                loss = SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
                run_eagerly= True)
        
        model.optimizer.lr.assign(1e-4)

        return model 

    def data_generator(self):
        """
            (WIP) A data generator would be far more efficient since I am using
            the same image over and over again with different text input and
            label
        """
        fnames = glob.glob('/'.join([self._data_dir, 'images','*']))
        with open('/'.join([self._base_dir, 'dataset', self._model_name, 'gauss_codes.txt'])) as fd:
            gauss_codes = fd.readlines()

    def predict(self, num, fnames=None):
        if fnames is None:
            fnames = glob.glob('/'.join([self._data_dir, 'images', '*.png']))

        for fname in fnames[:num]:
            image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (512,512))
            batch_image =image.reshape((1,512,512,1))

            output_seq =[self._to_num('[')]
            new_out = -1
            while (len(output_seq) < self._label_length) and ( new_out != self._to_num(']')): 
                rnn_input = pad_sequences([output_seq], maxlen=self._label_length)
                batch_rnn_input = np.array(rnn_input).reshape((1,self._label_length))
                rnn_output = self.model.predict(x = (batch_image, batch_rnn_input), verbose=0)
                new_out = np.argmax(rnn_output)
                output_seq.append(new_out)

            gauss_code = self._to_char(output_seq).numpy().tolist()
            gauss_code = ''.join([ char.decode('utf-8') for char in gauss_code])

            return gauss_code
