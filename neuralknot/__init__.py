#The AMD docker image does not export some modules as expected so need to do
#special imports. 
#FIXME Actually check if I am using the AMD docker package instead of just the
#same tensorflow version 
from tensorflow import __version__ as tf_version
AMD_CHECK = True if tf_version == '2.5.0' else False

