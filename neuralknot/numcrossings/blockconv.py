from tensorflow import reshape

from tensorflow.keras import Input
from tensorflow.keras.layers import  Layer
from tensorflow.keras import Model

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from neuralknot.numcrossings.utils import NumCrossings

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
    def __init__(self):
        self._model_name = 'blockconv'

        super().__init__()
        self._model_dir = '/'.join([self._base_dir, f'{self._model_name}_data'])
        
        self.model = self._make_model(len(self.class_names))

        self.training_sessions = super().get_training_sessions(self._model_dir)
        super().load_weights(self._model_dir, self.model)
        self.callbacks = super().update_callbacks(self._model_dir)

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
    
    def train_model(self, epochs):
        self.callbacks = super().update_callbacks(self._model_dir)
        super().train_model(epochs, self.model, self.callbacks)

    def plot_history(self):
        super().plot_history(self._model_dir)

    def plot_model(self):
        super().plot_model(self.model, self._model_dir)

    def evaluate_model(self):
        super().evaluate_model(self.model)

    def predict(self, num):
        super().predict(num, self.model)
