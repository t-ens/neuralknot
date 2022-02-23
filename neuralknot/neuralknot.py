import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd

from tensorflow.config.threading import set_intra_op_parallelism_threads
from tensorflow.config.threading import set_inter_op_parallelism_threads
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Softmax

import neuralknot.numcrossings.utils
import neuralknot.numcrossings.fullconv
import neuralknot.numcrossings.blockconv

import neuralknot.gaussencoder.simpleGRU

nets = {
    'numcrossings': {
        'data_loader': neuralknot.numcrossings.utils.load_data,
        'callbacks': neuralknot.numcrossings.utils.callbacks,
        'models': {
            'fullconv': neuralknot.numcrossings.fullconv,
            'blockconv': neuralknot.numcrossings.blockconv
        }
    },
    'gaussencode': {
        'models': {
            'simpleGRU': neuralknot.gaussencoder.simpleGRU
        }
    }
}

set_intra_op_parallelism_threads(8)
set_inter_op_parallelism_threads(8) 

def visualize_data(dataset, class_names, num=9):
    plt.figure()
    for images,labels in dataset.take(1):
        for i in range(num):
            ax = plt.subplot(3,3,i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy().astype("uint8")])
            plt.axis('off')
    plt.show()

def plot_history(dat):

    plt.figure()
    plt.plot(dat['loss'], 'b')
    plt.plot(dat['val_loss'], 'g')
    plt.legend(['loss', 'val_loss'])

    plt.figure()
    plt.plot(dat['accuracy'], 'b')
    plt.plot(dat['val_accuracy'], 'g')
    plt.legend(['acc', 'val_acc'])

    plt.show()

def predictions(num, ds, model, class_names):
    for images, labels in ds.take(1):    
        predictions = model.predict(images)    
        for i in range(num):    
    
            plt.figure()    
    
            plt.subplot(2,2,1)    
            plt.imshow(images[i].numpy().astype("uint8"))    
            plt.title(class_names[labels[i]])    
            plt.axis('off')    
     
            plt.subplot(2,2,2)    
            plt.bar(range(len(class_names)), predictions[i])    
     
            max_inds = np.argpartition(predictions[i], -3)[-3:]    
            max_ind = np.argmax(predictions[i])    
            for j in max_inds:    
                if j == max_ind:    
                    colour = 'r'    
                else:    
                    colour = 'g'    
                plt.text(j, predictions[i][j], class_names[j], c=colour, rotation='45', fontsize='x-small')    
                
            plt.show()
#
#
##history= model.fit(
##            train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE),
##            validation_data = val_ds.cache().prefetch(buffer_size=AUTOTUNE),
##            epochs = 100,
##            callbacks=callbacks)
#
#
#predict_model = Sequential([model, Softmax()])
#predictions(10, val_ds, predict_model, class_names)

def main():
    """Very simple interactive loop to load data and models, visualize either
    and train models
    """
    
    net_name = 'numcrossings'
    model_name = 'blockconv'

    base_dir = '/'.join(['neuralknot',net_name])
    data_dir = '/'.join([base_dir, 'dataset', 'images'])
    model_dir = '/'.join([base_dir, f'{model_name}_data'])
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    data_loaded = False
    model_loaded = False
    while True:
        print(f'Selected Model: {net_name}:{model_name}') 
        print('Options:')
        print('  1) Change model')
        print('  2) Plot model history')
        print('  3) Load data')
        print('  4) Visualize data')
        print('  5) Load model')
        print('  6) Plot model graph')
        print('  7) Train model')
        print('  8) Exit')
        choice = input('Enter number: ')

        if choice == '1':
            pass
        elif choice == '2':
            fname = '/'.join([model_dir, 'history.csv'])
            if os.path.isfile(fname):
                history = pd.read_csv(fname)
                plot_history(history)
            else:
                print('No training history found')
        elif choice == '3':
            if data_loaded:
                print('Data already loaded')
            else:
                train_ds, val_ds = nets[net_name]['data_loader'](data_dir)
                class_names = train_ds.class_names
                data_loaded = True
        elif choice == '4':
            if data_loaded:
                visualize_data(train_ds, class_names)
            else:
                print('No data loaded')
        elif choice == '5':
            if data_loaded:
                model = nets[net_name]['models'][model_name].make_model(len(class_names))
                model_loaded = True
                callbacks = nets[net_name]['callbacks'](model_dir)
                if os.path.isfile('/'.join([model_dir,'checkpoint'])):
                    model.load_weights(model_dir + '/')
                    #loss, acc = model.evaluate(val_ds, verbose=2)
                    #print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
            else:
                print('Load data first to determine number of classes.')
        elif choice == '6':
            fname = '/'.join([model_dir, 'model_graph.png'])
            plot_model(model, to_file=fname, show_shapes=True)
            plt.imshow(img.imread(fname))
            plt.axis('off')
            plt.show()
        elif choice == '7':
            pass
        elif choice == '8':
            return 0
