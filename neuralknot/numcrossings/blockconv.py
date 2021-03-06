import os

from tensorflow import reshape

from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Concatenate, Dense, Softmax

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from neuralknot.numcrossings.common import NumCrossings

class Blocks(Layer):
    """
    This layer breaks an image into a series of subimages
    """
    def __init__(self, block_size):
        super(Blocks, self).__init__()
        self.size = block_size

    def call(self, dat):
        batch_size = len(dat[:,0,0,0])
        num = int((512 / self.size)**2)
        return reshape(
                    dat,
                    (batch_size, self.size, self.size, num))

class BlockModel(NumCrossings):
    """
    This model counts the number of crossings by breaking the image into blocks
    of various sizes, doing convolution/pooling on each block then combining the
    result from each block in dense layers
    """
    def __init__(self):
        self._model_name = 'blockconv'
        self.__desc__ =  \
            """This model counts the number of crossings \n\
by breaking the image into blocks of various \n\
sizes, doing convolution/pooling on each \n\
block then combining the result from each \n\
block in dense layer"""


        super().__init__()
        self._model_dir = '/'.join([self._base_dir, f'{self._model_name}_data'])
        if not os.path.isdir(self._model_dir):
            os.mkdir(self._model_dir)
        
        self.model = self._make_model(len(self.class_names))
        self.predict_model = Sequential([self.model, Softmax()])
        self.load_weights()

    def _make_model(self, num_labels):
        inputs = Input(shape=((512,512,1)))
        x = Rescaling(1./255)(inputs)
        
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
        x = Dense(128, activation="relu")(x)
        outputs = Dense(num_labels)(x)
        
        model = Model(inputs, outputs, name="block_conv")
        
        model.compile(
                optimizer='adam',
                loss = SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                run_eagerly= True)
        
        model.optimizer.lr.assign(1e-4)
    
        return model
