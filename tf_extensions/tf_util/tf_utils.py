import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

# Custom imports
from .tf_constants import TF_PI

# Set global parameters for this file
K.set_floatx('float64')

__all__ = [
    "tf_gaussian_log_prob",
    "tf_gaussian_kld",
    "tf_gaussian_reparametrize",
    "tf_log_mean_exp",
    "tfp_sample_beta_distribution",
    "tf_sample_beta_distribution",
    "spot_mix",
    "lerp_mix",
    "patch_mix",
    "interp_loss_weight"
]


def tf_gaussian_log_prob(z, mu, logvar):
    return -0.5*(tf.math.log(2.0*TF_PI) + logvar + tf.math.pow((z-mu), 2.0)/tf.math.exp(logvar))


def tf_gaussian_kld(z_mu, z_logvar):
    return 0.5*(K.pow(z_mu, 2.0) + K.exp(z_logvar) - 1.0 - z_logvar)


def tf_gaussian_reparametrize(mu, logvar):
    std = K.exp(0.5 * logvar)
    return mu + tf.random.normal(tf.shape(std), dtype=K.floatx()) * std


def tf_log_mean_exp(x, axis):
    m  = tf.math.reduce_max(x, axis=axis)
    m2 = tf.math.reduce_max(x, axis=axis, keepdims=True)
    return m + K.log(K.mean(K.exp(x-m2), axis))


def tfp_sample_beta_distribution(alpha, beta, shape=[1]):
    '''
        Draws n samples k from the Beta Distribution.
        This version of the function might not work on GPU.
    '''
    dist = tfp.distributions
    beta_dist = dist.Beta(alpha, beta)
    k_ = beta_dist.sample(shape)
    return tf.cast(k_, K.floatx())


def tf_sample_beta_distribution(alpha, beta, shape=[1]):
    '''
        Draws n samples k from the Beta Distribution.
        Here, we compute a sample through two Gamma distributions.
        This version of the function is GPU enabled.
    '''
    gamma_1_sample = tf.random.gamma(shape=shape, alpha=alpha, dtype=K.floatx())
    gamma_2_sample = tf.random.gamma(shape=shape, alpha=beta , dtype=K.floatx())
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample + K.epsilon())


@tf.autograph.experimental.do_not_convert
def _mask_with_ratio(shape, ratio):
    uniform = tf.random.uniform
    logical_mask = uniform(shape, dtype=K.floatx()) < ratio
    return logical_mask


@tf.autograph.experimental.do_not_convert
def spot_mix(l0, l1, k_):
    layer_shape = tf.shape(l0)
    m1    = _mask_with_ratio(layer_shape, k_)
    m1_f  = tf.cast(m1, K.floatx())
    elem0 = tf.math.multiply(m1_f, l0)
    elem1 = tf.math.multiply((1. - m1_f), l1)
    return elem0, elem1


@tf.autograph.experimental.do_not_convert
def lerp_mix(l0, l1, k_):
    elem0 = tf.math.multiply(k_, l0)
    elem1 = tf.math.multiply(1. - k_, l1)
    return elem0, elem1


@tf.autograph.experimental.do_not_convert
def _setup_patch_mix(layer_shape, k_):
    vsize = tf.cast(layer_shape[1], dtype=K.floatx())
    patch_size = tf.math.floor(vsize * k_)
    patch_size_half = patch_size * .5
    rng = tf.random.uniform(shape=[1], dtype=K.floatx())
    offset = tf.math.floor(rng * vsize)
    ix = tf.squeeze(tf.cast(offset, dtype=tf.int32))
    lo = tf.cast(patch_size_half, dtype=tf.int32)
    hi = tf.cast(tf.math.ceil(patch_size_half), dtype=tf.int32)
    sz = tf.cast(vsize, dtype=tf.int32)    
    indeces = tf.range(start=0, limit=sz, delta=1, dtype=tf.int32)
    return ix, lo, hi, sz, indeces

@tf.autograph.experimental.do_not_convert
def _handle_lower_bound(ix, lo, hi, sz, indeces):
    e = sz + (ix-lo)
    _find = indeces >= e
    patch_mask = tf.where(_find, 1, 0)
    _find = indeces < (ix + hi)
    patch_mask += tf.where(_find, 1, 0)
    return patch_mask

@tf.autograph.experimental.do_not_convert
def _handle_upper_bound(ix, lo, hi, sz, indeces):
    e = (ix+hi) - sz
    _find = indeces <= e
    patch_mask = tf.where(_find, 1, 0)
    _find = indeces > (ix - lo)
    patch_mask += tf.where(_find, 1, 0)
    return patch_mask

@tf.autograph.experimental.do_not_convert
def _patch_middle(ix, lo, hi, sz, indeces):
    _find0 = indeces >= (ix - lo)
    _find1 = indeces <  (ix + hi)
    _find  = tf.logical_and(_find0, _find1)
    patch_mask = tf.where(_find, 1, 0)
    return patch_mask

@tf.autograph.experimental.do_not_convert
def patch_mix(l0, l1, k_):
    layer_shape = tf.shape(l0)
    ix, lo, hi, sz, indeces = _setup_patch_mix(layer_shape, k_)
        
    if (ix - lo) < 0:
        patch_mask = _handle_lower_bound(ix, lo, hi, sz, indeces)
    elif (ix + hi) > sz:
        patch_mask = _handle_upper_bound(ix, lo, hi, sz, indeces)
    else:
        patch_mask = _patch_middle(ix, lo, hi, sz, indeces)
    
    shape_carrier = tf.zeros(shape=layer_shape, dtype=K.floatx())
    mask0 = shape_carrier + tf.cast(patch_mask, dtype=K.floatx())
    mask1 = tf.ones_like(mask0) - mask0
    elem0 = tf.math.multiply(mask0, l0)
    elem1 = tf.math.multiply(mask1, l1)
    return elem0, elem1

@tf.autograph.experimental.do_not_convert
def interp_loss_weight(k, r=3.):
    p = 1. / r
    q0 = tf.math.pow(k, p)
    q1 = tf.math.pow(1. - k, p)
    return 2. * tf.math.divide(q0, q0 + q1)
    
    
    