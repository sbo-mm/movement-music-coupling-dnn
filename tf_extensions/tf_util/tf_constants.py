import tensorflow as tf
import tensorflow.keras.backend as K

K.set_floatx('float64')
__BASE_CPU_PI   = 3.14159265
__BASE_CPU_LOG2 = 0.69314718056

__all__ = [
    "TF_PI",
    "TF_LOG2"
]

with tf.name_scope(__name__):
    TF_PI = tf.constant(__BASE_CPU_PI, 
                        dtype=K.floatx(), 
                        name="pi")
    
    TF_LOG2 = tf.constant(__BASE_CPU_LOG2,
                         dtype=K.floatx(),
                         name="log2")