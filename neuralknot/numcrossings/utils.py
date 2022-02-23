import glob

from tensorflow.keras.preprocessing import image_dataset_from_directory
#AMD tensorflow docker image doesn't export image_dataset_from_directoy like    
#usual    

from tensorflow.data import AUTOTUNE

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

def load_data(folder, vsplit=0.2, image_size=(512,512), batch_size=32):    
    
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

def newest_checkpoint(save_dir):
    sessions = glob.glob('/'.join([save_dir, 'session_*']))

    nums = []
    for session in sessions:
        num = 0
        for i, char in enumerate(reversed(session)):
            if char == '_':
                break
            else:
                num += 1
        nums.append(int(session[-num:]))
    return max(nums)


def callbacks(save_dir):
    checkpoint = ModelCheckpoint(      
        filepath = save_dir,      
        save_weights_only=True,      
        verbose = 1)

    history_logger = CSVLogger(    
        '/'.join([save_dir,'history.csv']),    
        separator = ',',    
        append=True)

    return [checkpoint, history_logger]
