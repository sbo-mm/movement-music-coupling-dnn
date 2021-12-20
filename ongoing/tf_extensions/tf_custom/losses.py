import tensorflow as tf
from tensorflow.keras import backend as K

# Custom imports
from .. import tf_util 

# Set global parameters for this file
K.set_floatx('float64')

all = [
    "SpectralPhaseLogProbLoss",
    "SpectralAmplitudeLogProbLoss",
    "SpectralPhaseLoss"
]


class _SpectralLogProbLoss(tf.keras.losses.Loss):

    def __init__(self, inverse_normalize=None, *args, **kwargs):
        super(_SpectralLogProbLoss, self).__init__(*args, **kwargs)
        self.log_prob_fn = None
        self.inverse_normalize = inverse_normalize
    
    def call(self, y_true, y_pred):
        if self.inverse_normalize:
            y_true = self.inverse_normalize(y_true)
            y_pred = self.inverse_normalize(y_pred)
               
        log_prob_batch = self.log_prob_fn(
            y_true, y_pred, tf.ones_like(y_pred, dtype=K.floatx()))
        log_prob_batch = tf.math.reduce_sum(log_prob_batch, axis=-1)
        return tf.math.reduce_mean(log_prob_batch, axis=0)

    
class SpectralPhaseLogProbLoss(_SpectralLogProbLoss):
    
    def __init__(self, inverse_normalize=None, name="phase_log_prob_loss", *args, **kwargs):
        super(SpectralPhaseLogProbLoss, self).__init__(inverse_normalize=inverse_normalize, 
                                                name=name, *args, **kwargs)
        self.log_prob_fn = tf_util.tf_vonmises_log_prob
    
class SpectralAmplitudeLogProbLoss(_SpectralLogProbLoss):
    
    def __init__(self, inverse_normalize=None, name="amplitude_log_prob_loss", *args, **kwargs):
        super(SpectralAmplitudeLogProbLoss, self).__init__(inverse_normalize=inverse_normalize, 
                                                           name=name, *args, **kwargs)
        self.log_prob_fn = tf_util.tf_gauss_log_prob

        
class SpectralPhaseLoss(tf.keras.losses.Loss):
    
    def __init__(self, inverse_normalize=None, name="phase_loss", *args, **kwargs):
        super(SpectralPhaseLoss, self).__init__(name=name, *args, **kwargs)
        self.inverse_normalize = inverse_normalize
        self.cpx = tf.dtypes.complex
    
    def phase_spectral_loss(self, y_true, y_pred):
        err_raw = y_true - y_pred
        err_cpx = self.cpx(tf.zeros_like(err_raw), err_raw)
        err_exp = tf.math.exp(err_cpx)
        err_abs = tf.math.abs(1. - err_exp)
        err_pow = tf.math.pow(err_abs, 2)
        ps_loss = .5 * err_pow
        return ps_loss
    
    def call(self, y_true, y_pred):
        if self.inverse_normalize:
            y_true = self.inverse_normalize(y_true)
            y_pred = self.inverse_normalize(y_pred)
        
        phase_loss_batch = self.phase_spectral_loss(y_true, y_pred)
        phase_loss = tf.math.reduce_sum(phase_loss_batch, axis=-1)
        return tf.math.reduce_mean(phase_loss, axis=0)