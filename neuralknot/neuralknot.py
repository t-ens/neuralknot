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
#
#

def set_directories(net_name, model_name):
    bd = '/'.join(['neuralknot',net_name])
    dd = '/'.join([bd, 'dataset', 'images'])
    md = '/'.join([bd, f'{model_name}_data'])
    if not os.path.isdir(md):
        os.mkdir(md)

    return bd, dd, md

def load_data(net_name, data_dir):
    train_ds, val_ds = nets[net_name]['data_loader'](data_dir)
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

def main():
    """Very simple interactive loop to load data and models, visualize either
    and train models
    """
    
    net_name = 'numcrossings'
    model_name = 'blockconv'

    base_dir, data_dir, model_dir = set_directories(net_name, model_name)

    data_loaded, model_loaded = False, False
    while True:
        print(f'Selected Model: {net_name}:{model_name}') 
        print('Options:')
        print('  1) Change model')
        print('  2) Plot model history')
        print('  3) Visualize data')
        print('  4) Load model')
        print('  5) Plot model graph')
        print('  6) Evaluate model')
        print('  7) Train model')
        print('  8) Predict with model')
        print('  9) Exit')
        choice = input('Enter number: ')

        if choice == '1':
            print(f'Available networks: {[ key for key in nets.keys()]}')
            netchoice = input('Choose net: ')
            
            if netchoice in nets:
                print(f'Available models: {[key for key in nets[netchoice]["models"].keys()]}')
                modelchoice = input('Choose model: ')
                if modelchoice in nets[netchoice]['models']:
                    net_name = netchoice
                    model_name = modelchoice
                    base_dir, data_dir, model_dir = set_directories(net_name, model_name)
                else:
                    print('Invalid choice')
            else:
                print('Invalid choice')


        elif choice == '2':
            fname = '/'.join([model_dir, 'history.csv'])
            if os.path.isfile(fname):
                history = pd.read_csv(fname)
                plot_history(history)
            else:
                print('No training history found')

        elif choice == '3':
            if not data_loaded:
                train_ds, val_ds, class_names = load_data(net_name, data_dir)
                data_loaded = True
            visualize_data(train_ds, class_names)

        elif choice == '4':
            if not data_loaded:
                train_ds, val_ds, class_names = load_data(net_name, data_dir)
                data_loaded = True

            model = nets[net_name]['models'][model_name].make_model(len(class_names))
            model_loaded = True
            callbacks = nets[net_name]['callbacks'](model_dir)
            if os.path.isfile('/'.join([model_dir,'checkpoint'])):
                model.load_weights(model_dir + '/')

        elif choice == '5':
            fname = '/'.join([model_dir, 'model_graph.png'])
            plot_model(model, to_file=fname, show_shapes=True)
            plt.imshow(img.imread(fname))
            plt.axis('off')
            plt.show()

        elif choice == '6':
            if data_loaded and model_loaded:
                loss, acc = model.evaluate(val_ds, verbose=2)
                print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
            else:
                print('No model loaded')

        elif choice == '7':
            epochs = input('How many epochs?')
            history= model.fit(
                train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE),
                validation_data = val_ds.cache().prefetch(buffer_size=AUTOTUNE),
                epochs = epochs,
            callbacks=callbacks)

        elif choice == '8':
            num_predictions = input('How many predictions (max 32): ')
            valid_input = True
            try:
                num_predictions = int(num_predictions)
            except ValueError:
                valid_input = False

            if  valid_input and 0 <= num_predictions <= 32:
                predict_model = Sequential([model, Softmax()])
                predictions(num_predictions, val_ds, predict_model, class_names)
            else:
                print('Invalid input')

        elif choice == '9' or 'q':
            return 0
