import os

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from neuralknot.numcrossings.utils import NumCrossings

class FullConv(NumCrossings):
    def __init__(self):
        self._model_name = 'fullconv'

        super().__init__()
        self._model_dir = '/'.join([self._base_dir, f'{self._model_name}_data'])
        if not os.path.isdir(self._model_dir):
            os.mkdir(self._model_dir)
        
        self.model = self._make_model(len(self.class_names))

        self.training_sessions = super().get_training_sessions(self._model_dir)
        super().load_weights(self._model_dir, self.model)
        self.callbacks = super().update_callbacks(self._model_dir)


    def _make_model(self, num_labels):

        model = Sequential([                                        
            Rescaling(1./255, input_shape=(512,512,1)),    
            Conv2D(32, (3,3), activation='relu'),                   
            MaxPool2D((2,2)),                                               
            Conv2D(32, (3,3), activation='relu'),                   
            MaxPool2D((2,2)),                                       
            Conv2D(32, (3,3), activation='relu'),    
            MaxPool2D((2,2)),                                       
            Conv2D(64, (3,3), activation='relu'),                   
            MaxPool2D((2,2)),                                       
            Conv2D(64, (3,3), activation='relu'),                   
            MaxPool2D((2,2)),                                       
            Conv2D(128, (3,3), activation='relu'),                  
            MaxPool2D((2,2)),                                       
            Conv2D(128, (3,3), activation='relu'),                  
            MaxPool2D((2,2)),                                       
            Flatten(),                                              
            Dense(128, activation='relu'),                          
            Dense(num_labels)                                       
            ])                                                                      

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
            
