import tensorflow as tf
import tensorflow.keras.backend as K

K.set_floatx('float64')
__BASE_CPU_PI = 3.14159265

__all__ = [
    "TF_PI"
]

with tf.name_scope(__name__):
    TF_PI = tf.constant(__BASE_CPU_PI, 
                        dtype=K.floatx(), 
                        name="pi")
    
