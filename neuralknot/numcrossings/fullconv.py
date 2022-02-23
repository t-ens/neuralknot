from tensorflow.keras.models import Sequential

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import SparseCategoricalCrossentropy

def make_model(num_labels):

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
            
