from numpy import reshape, argmax, argpartition
import matplotlib.pyplot as plt

from tensorflow.data import AUTOTUNE

from neuralknot.common import KnotModel

#AMD docker image specific imports#############################################
try:
    from tensorflow.keras.utils import image_dataset_from_directory
except (ImportError, ModuleNotFoundError):
    from tensorflow.keras.preprocessing import image_dataset_from_directory
###############################################################################


class NumCrossings(KnotModel):
    def __init__(self):
        self._net_name = 'numcrossings'

        self._base_dir = '/'.join(['neuralknot', self._net_name])
        self._data_dir = '/'.join([self._base_dir, 'dataset', 'images'])

        self.train_ds, self.val_ds = self.load_data(self._data_dir)
        self.class_names = self.train_ds.class_names

    def load_data(self, folder, vsplit=0.2, image_size=(512,512), batch_size=32):    
        
        train_ds = image_dataset_from_directory(    
                    folder,    
                    seed = 123,    
                    validation_split = vsplit,    
                    subset='training',    
                    image_size=image_size,    
                    batch_size=batch_size,    
                    color_mode='grayscale')    
        
        val_ds = image_dataset_from_directory(    
                    folder,    
                    seed = 123,    
                    validation_split = vsplit,    
                    subset='training',    
                    image_size=image_size,    
                    batch_size=batch_size,    
                    color_mode='grayscale')    
        
        return train_ds, val_ds 

    def predict(self, image, axes):
        image = reshape(image, (1, *image.shape, 1))
        predictions = self.predict_model.predict(image)
       
        axes[0].cla()
        axes[0].imshow(image[0])    
        axes[0].set_title('Input image')
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)

        axes[1].cla()
        axes[1].set_title('Prediction Probabilities')
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)
        for i,_ in enumerate(predictions):
            axes[1].bar(range(1,len(self.class_names)+1), predictions[i])    
       
            max_inds = argpartition(predictions[i], -3)[-3:]    
            max_ind = argmax(predictions[i])    
            for j in max_inds:    
                if j == max_ind:    
                    colour = 'r'    
                else:    
                    colour = 'g'    
                axes[1].text(
                        j+1, 
                        predictions[i][j], 
                        str(j+1), 
                        c=colour, 
                        rotation='45', 
                        fontsize='x-small')    

    def visualize_data(self, axes, num=9):
        for images,labels in self.train_ds.take(1):
            for i in range(num):
                axes[i].imshow(images[i].numpy().astype("uint8"))
                axes[i].set_title(self.class_names[labels[i].numpy().astype("uint8")])
                axes[i].axes.get_xaxis().set_visible(False)
                axes[i].axes.get_yaxis().set_visible(False)
