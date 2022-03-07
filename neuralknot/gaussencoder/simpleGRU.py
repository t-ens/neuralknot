import os
import glob
import numpy as np

import matplotlib.pyplot as plt
import cv2

from neuralknot.numcrossings.blockconv import Blocks 

from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.strings import unicode_split

from tensorflow.keras import Input
from tensorflow.keras.layers import  Layer
from tensorflow.keras import Model

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Concatenate, Dense, GRU, Embedding, Add

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences

from neuralknot.gaussencoder.common import GaussEncoder

#AMD docker image specific imports#############################################
from neuralknot import AMD_CHECK
if AMD_CHECK:
    from tensorflow.keras.preprocessing import image_dataset_from_directory
    from tensorflow.python.keras.layers.preprocessing.string_lookup import StringLookup
else:
    from tensorflow.keras.utils import image_dataset_from_directory
    from tensorflow.keras.layers import StringLookup
###############################################################################

class SimpleGRU(GaussEncoder):
    """
        This model follows the approach described in the paper "Show and Tell: A
        Neural Image Caption Generator" to convert an image to a sequence. An
        element of the dataset is an image and a partial Gauss code. Each such
        datapoint is labelled by the next character in the Gauss code.
        Convolution layers are performed on the image, and the resulting vector
        initializes the hidden state of the GRU layer. The partial Gauss code is
        then fed through the GRU layers to predict the output character. 
    """
    def __init__(self):
        self._model_name = 'simplegru'
        self.__desc__ =  \
            """This model computes the Gauss code of \n\
a knot diagram by performing blockwise \n\
convolution/pooling then feeding the  \n\
result as the initial state to GRU cells \n\
which predict the next character in the \n\
Gauss code thought of as a string"""

        super().__init__()
        self._model_dir = '/'.join([self._base_dir, f'{self._model_name}_data'])
        if not os.path.isdir(self._model_dir):
            os.mkdir(self._model_dir)

        self._data_dir = '/'.join([self._base_dir, 'dataset2'])
        # '@' is a padding character. '[' marks the beginning of a gauss code
        # and ']' marks the end. Everything else is either an integer, a minus
        # sign or a comma
        #This will allow knots with arbitrarily many crossings after training
        #the model on some finite set of knots. The model will probably work
        #better if the dictionary consists of of integers themselves rather than
        #individual digits
        self._dictionary = ['@', '-',',', '[',']'] + [ str(i) for i in range(10)]
        self._to_num = StringLookup(vocabulary=self._dictionary, oov_token="")
        self._to_char = StringLookup(vocabulary=self._to_num.get_vocabulary(), oov_token="", invert=True)

        self.train_ds, self.val_ds = self.load_data()
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

    def _parse_gauss_string(self, w):
        match = re.search( '.*?\[(.*)\]', w)
        code = match.groups(0)[0].replace(' ', '')
        return code

    def load_data(self, vsplit=0.2, image_size=(512,512), batch_size=32):
        """
            Load data from gaussencode/dataset. This is the wrong way of doing
            it since I am storing the same image file many time which will lead
            to massive datasets. This should be replaced with a data generator
            or something which will generate all the appropriate labellings from
            a single image instead
        """
        image_dir = '/'.join([self._data_dir, 'images'])

        with open('/'.join([self._data_dir, 'text_input.txt']), 'r') as fd:
            text_inputs = fd.readlines()
        with open('/'.join([self._data_dir, 'labels.txt']), 'r') as fd:
            labels = fd.readlines()

        #Remove spaces and newline characters
        text_inputs = [ line.replace(' ','') for line in text_inputs ]
        text_inputs = [ line.rstrip() for line in text_inputs ]
        #Encode using dictionary 
        text_inputs = [ self._to_num([ch for ch in line]) for line in text_inputs ]
        #Pad to length of longest input
        text_inputs = pad_sequences(text_inputs)
        self._label_length = text_inputs[0].shape[0]
        text_ds = Dataset.from_tensor_slices(text_inputs)

        labels= [ line.rstrip() for line in labels ]
        labels  = [ self._to_num(label) for label in labels ]
        label_ds = Dataset.from_tensor_slices(labels)

        fnames = glob.glob('/'.join([image_dir, '*']))
        num_images = len(fnames)
        images = image_dataset_from_directory(
                self._data_dir,
                labels=None,
                shuffle=False,
                image_size=image_size,
                batch_size=None,
                color_mode='grayscale')

        input_ds = Dataset.zip((images, text_ds))
        full_ds = Dataset.zip((input_ds, label_ds))

        val_num = round(num_images * vsplit)
        val_ds = (full_ds.take(val_num)
                    .padded_batch(32)
                    .prefetch(buffer_size=AUTOTUNE))
        train_ds = (full_ds.skip(val_num)
                    .padded_batch(32)
                    .prefetch(buffer_size=AUTOTUNE))

        return train_ds, val_ds

    def visualize_data(self, axes, num=9):
        for (images, text), labels in self.train_ds.take(1):
            for i in range(num):
                gauss_code = ''.join([ char.decode('utf-8') for char in self._to_char(text[i]).numpy().tolist()])
                label = self._to_char(labels[i]).numpy().decode('utf-8')

                axes[i].imshow(images[i].numpy().astype("uint8"))
                axes[i].set_title(gauss_code, fontsize=7)
                axes[i].axes.get_xaxis().set_visible(False)
                axes[i].axes.get_yaxis().set_visible(False)
                axes[i].text(180, 560, f'Label: {label}')
