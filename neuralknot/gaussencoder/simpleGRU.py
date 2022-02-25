"""Feed the inputs from blockwise convolution into GRU nodes.
Gauss codes can be read off a knot diagram by starting anywhere, traversing the
knot and recording crossings. Need the previously visited crossings to determine
the next crossings to visit so RNN is required.
"""
from neuralknot.numcrossings.blockconv import Blocks 

from tensorflow.keras import Input
from tensorflow.keras.layers import  Layer
from tensorflow.keras import Model

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from neuralknot.gaussencoder.utils import DetectCrossings


class SimpleGRU(DetectCrossings):
    def _make_model(num_labels):
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
    
        x = GRU(10, dropout=0.1)(x)
    
        #Since I want to allow knot diagrams with an arbitrary number of arcs, I
        #will output the Gauss code as a string so that I only need the finitely
        #many (12) symbols {',','-','0','1','2','3','4','5','6','7','8','9'}
        x = Dense(12)(x)
        
        model = Model(inputs, outputs, name="block_conv")
        
        model.compile(
                optimizer='adam',
                loss = SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                run_eagerly= True)
        
        model.optimizer.lr.assign(1e-4)

        return model
    
