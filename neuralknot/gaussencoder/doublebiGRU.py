import os
import glob

from tensorflow.keras import Input
from tensorflow.keras.layers import  Layer
from tensorflow.keras import Model

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import (Conv2D, MaxPool3D, Flatten, Concatenate, 
    Dense, GRU, Embedding, Add, Reshape, Permute, Bidirectional, Softmax)

from tensorflow import cast, shape, ones    
from tensorflow.nn import ctc_loss as nn_ctc_loss

from neuralknot.gaussencoder.common import GaussEncoder

def ctc_loss(y_t, y_p):    
    """
        Compute input length and label length then feed to ctc_batch_cast
        function
    """
    y_t += 101
    y_t = y_t.to_tensor(0)
    batch_length = cast(shape(y_t)[0], dtype="int64")    
    logit_length = cast(shape(y_p)[1], dtype="int64")    
    label_length = cast(shape(y_t)[1], dtype="int64")    
     
    logit_length = logit_length * ones(shape=(batch_length), dtype="int64")    
    label_length = label_length * ones(shape=(batch_length), dtype="int64")    
     
    #loss = ctc_batch_cost(y_t, y_p, input_length, label_length)    
    loss = nn_ctc_loss(y_t, y_p, logit_length, label_length,
            logits_time_major=False)
    return loss

class DoubleBiGRU(GaussEncoder):
    """Compute Gauss codes retaining geometrical position of features.
        
    Break image into horizontal and vertical layers. Convolve each layer and
    input the sequence of horizontal layers into bidirectional GRU cells, and
    separately feed the sequence of vertical layers into bidirectional GRU
    cells. Combine outputs and use CTC loss to generate output sequence.
    """
    def __init__(self, image_size=(512,512), width=16, height=16, max_crossings=100):
        """
            image_size: size of input image
            width: width of vertical strips
            height: height of horizontal strips
            max_crossings: max number of crossings a knot can have

            width must evenly divide image_size[0] and height must evenly divide image_size[1]
        """
        self._model_name = 'doublebigru'
        self.__desc__ = \
            """This model computes the Gauss code of \n\
a knot diagram by creating a sequence of \n\
horizontal and a sequence of vertical \n\
layers, doing convolution/pooling in \n\
each layer, running the two sequences \n\
through bidirectional GRU layers, and \n\
adding the two sequences. CTC loss is \n\
is used"""

        super().__init__()
        self._model_dir = '/'.join([self._base_dir, f'{self._model_name}_data'])
        if not os.path.isdir(self._model_dir):
            os.mkdir(self._model_dir)

        self.width = width
        self.height = height
        self.image_size = image_size
        self.max_crossings = max_crossings
        
        self.train_ds, self.val_ds = self.load_data()
        self.model = self._make_model()
        self.load_weights()


    def _make_model(self):
        image_input = Input(shape=(512,512,1), batch_size=32)
        x = Rescaling(1./255)(image_input)

        #Break image vertically, convolve then feed into GRU
        x1 = Reshape((512//self.width, self.width, 512, 1))(x)
        x1 = Conv2D(32, (2,4), input_shape=image_input[2:])(x1)
        x1 = MaxPool3D(pool_size=(1,2,4))(x1)
        x1 = Conv2D(32, (2,4), input_shape=image_input[2:])(x1)
        x1 = MaxPool3D(pool_size=(1,2,4))(x1)
        x1 = Reshape((32, x1.get_shape()[2]*x1.get_shape()[3]*x1.get_shape()[4]))(x1)
        x1 = Bidirectional(GRU(100, dropout=0.2, return_sequences=True))(x1)
        x1 = Dense(2*self.max_crossings+1)(x1)

        #Break image horizontally, convolve then feed into GRU
        x2 = Permute((2,1,3))(x)
        x2 = Reshape((512//self.height, self.height, 512, 1))(x2)
        x2 = Permute((1,3,2,4))(x2)
        x2 = Conv2D(32, (4,2), input_shape=image_input[2:])(x2)
        x2 = MaxPool3D(pool_size=(1,4,2))(x2)
        x2 = Conv2D(32, (4,2), input_shape=image_input[2:])(x2)
        x2 = MaxPool3D(pool_size=(1,4,2))(x2)
        x2 = Reshape((32, x2.get_shape()[2]*x2.get_shape()[3]*x2.get_shape()[4]))(x2)
        x2 = Bidirectional(GRU(100, dropout=0.2, return_sequences=True))(x2)
        x2 = Dense(2*self.max_crossings+1)(x2)

        x = Add()([x1, x2])
        output = Softmax()(x)

        model = Model(image_input, x)
        model.compile(
                optimizer='adam',
                loss = ctc_loss,
                metrics=[],
                run_eagerly= True)
        model.optimizer.lr.assign(1e-4)

        return model
