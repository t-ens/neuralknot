import glob
import re

import matplotlib.pyplot as plt

from tensorflow.data import Dataset, AUTOTUNE

from tensorflow.ragged import constant
from tensorflow.keras.preprocessing.sequence import pad_sequences

from neuralknot.common import KnotModel

#AMD docker image specific imports#############################################
try:
    from tensorflow.keras.utils import image_dataset_from_directory
except (ImportError, ModuleNotFoundError):
    from tensorflow.keras.preprocessing import image_dataset_from_directory
###############################################################################


class GaussEncoder(KnotModel):
    """
        This class contains methods and objects common to all models for gauss
        encoding. Do not instantiate this class directly, you should always
        instantiate the model class.
    """
    def __init__(self):
        self._net_name = 'gaussencoder'
        
        self._base_dir = '/'.join(['neuralknot', self._net_name])
        self._data_dir = '/'.join([self._base_dir, 'dataset'])


    def load_data(self, vsplit=0.2, image_size=(512,512)):
        """
            Load data from gaussencode/dataset. This is the wrong way of doing
            it since I am storing the same image file many time which will lead
            to massive datasets. This should be replaced with a data generator
            or something which will generate all the appropriate labellings from
            a single image instead
        """
        image_dir = '/'.join([self._data_dir, 'images'])

        with open('/'.join([self._data_dir, 'labels.txt']), 'r') as fd:
            labels = fd.readlines()
        num_datapoints = len(labels)

        labels= [ line.rstrip() for line in labels ]
        labels  = [ [int(num) for num in label.split(',')] for label in labels ]
        labels = constant(labels)
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

        full_ds = Dataset.zip((images, label_ds))
        full_ds = full_ds.shuffle(num_datapoints)

        val_num = round(num_images * vsplit)
        val_ds = (full_ds.take(val_num)
                    .batch(self.batch_size)
                    .prefetch(buffer_size=AUTOTUNE))
        train_ds = (full_ds.skip(val_num)
                    .batch(self.batch_size)
                    .prefetch(buffer_size=AUTOTUNE))
        
        return train_ds, val_ds

    def visualize_data(self, axes, num=9):
        num_takes = num // self.batch_size + 1
        num_per_take = [ self.batch_size for _ in range(num_takes)]
        num_per_take[-1] = num % self.batch_size 
        for j, (images, label) in enumerate(self.train_ds.take(num_takes)):
            for i in range(num_per_take[j]):
                gauss_code = '[' + ','.join([str(num) for num in label[i].numpy()]) + ']'

                axes[j*self.batch_size + i].imshow(images[i].numpy().astype("uint8"))
                axes[j*self.batch_size + i].set_title(gauss_code, fontsize=7)
                axes[j*self.batch_size + i].axes.get_xaxis().set_visible(False)
                axes[j*self.batch_size + i].axes.get_yaxis().set_visible(False)

    def predict(self, image, axes):
        pass
