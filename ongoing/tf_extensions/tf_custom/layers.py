import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from tensorflow.keras import backend as K

# Custom imports
from .. import tf_util 

# Set global parameters for this file
K.set_floatx('float64')

__all__ = [
    "MixingBlock",
    "Snake"
]

class MixingBlock(tf.keras.layers.Layer):
    
    def __init__(self, alpha=2, const=True, kval=0.5, mix_type="spot", *args, **kwargs):
        super(MixingBlock, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.const = const
        self.kval = kval
        self.mix_type = mix_type
        
    def build(self, input_shape):
        self.concentration = tf.constant(self.alpha, dtype=K.floatx())
        self.k_ = tf.constant(self.kval, dtype=K.floatx())
        super(MixingBlock, self).build(input_shape)
        
    def call(self, inputs, training=True):
        l0, l1 = inputs
        k_ = None
        if not self.const:
            a  = self.concentration
            k_ = tf_util.tfp_sample_beta_distribution(a, a, shape=[1])
        else:
            k_ = self.k_
        
        if self.mix_type == "patch":
            mask0, mask1 = tf_util.patch_mix(l0, l1, k_)
        elif self.mix_type == "lerp":
            mask0, mask1 = tf_util.lerp_mix(l0, l1, k_)
        else:
            mask0, mask1 = tf_util.spot_mix(l0, l1, k_)
        
        cutmix = 2. * (mask0 + mask1)
        return cutmix, k_
        
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    
class Snake(tf.keras.layers.Layer):
    
    def __init__(self, alpha=.5, trainable=False, **kwargs):
        super(Snake, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha
        self.trainable = trainable
        
    def build(self, input_shape):
        self.alpha_factor = tf.Variable(
            self.alpha, dtype=K.floatx(), name='alpha_factor')
        
        if self.trainable:
            self._trainable_weights.append(self.alpha_factor)
        
        super(Snake, self).build(input_shape)

    def call(self, inputs, training=True, mask=None):
        return tfa.activations.snake(inputs, self.alpha_factor)

    def compute_output_shape(self, input_shape):
        return input_shape