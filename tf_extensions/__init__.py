# Set top level globals for all submodules
from tensorflow.keras import backend as KERAS_BACKEND
KERAS_BACKEND.set_floatx('float32')

# Import top level submodules
from . import tf_custom
from . import tf_util