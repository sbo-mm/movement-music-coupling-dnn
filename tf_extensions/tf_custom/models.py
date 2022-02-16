import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# Set global parameters for this file
K.set_floatx('float64')

# Custom imports
from .. import tf_util 
from .. tf_util import TF_PI, TF_LOG2


__all__ = [
    "GaussianBetaVAE"
]


def abstractmethod(method):
    """
    An @abstractmethod member fn decorator.
    """
    def default_abstract_method(*args, **kwargs):
        raise NotImplementedError('call to abstract method ' + repr(method))
    default_abstract_method.__name__ = method.__name__    
    return default_abstract_method


class BaseVAE(tf.keras.models.Model):
    
    def __init__(self, input_dim, latent_dim, create_encoder_func, create_decoder_func, *args, **kwargs):
        super(BaseVAE, self).__init__(*args, **kwargs)
        self.custom_optimizer = None
        self.latent_dim = latent_dim
        self.input_dim  = input_dim
        self.encoder = create_encoder_func(input_dim, latent_dim)
        self.decoder = create_decoder_func(input_dim, latent_dim)
        
        self.build_data = tf.convert_to_tensor(
            np.zeros((1, np.prod(input_dim))), dtype=K.floatx(), name="build_data"
        )
        
    def custom_compile(self, optimizer, *args, **kwargs):
        # Optimizer is parsed from call to ´custom_compile´
        self.custom_optimizer = optimizer
       
        # Define some metric data displays
        self.elbo_loss = tf.keras.metrics.Mean(name="ELBO")
        self.reg_loss  = tf.keras.metrics.Mean(name="reg")
        self.rec_loss  = tf.keras.metrics.Mean(name="rec")
        
        # Setup model for weight saving and summaries
        self(self.build_data, training=False)
        
        # Call base-class compile with extra args if necessary
        super(BaseVAE, self).compile(*args, **kwargs)
        
    @property
    def metrics(self):
        return [self.elbo_loss, self.reg_loss, self.rec_loss]
        
    @tf.autograph.experimental.do_not_convert
    def encode(self, x, training=True):
        z_params = self.encoder(x, training=training)
        z_mu, z_logvar = tf.split(z_params, 2, 1)
        return z_mu, z_logvar
    
    @tf.autograph.experimental.do_not_convert
    def decode(self, z, training=True):
        x_params = self.decoder(z, training=training)
        x_mu, x_logvar = tf.split(x_params, 2, 1)
        return x_mu, x_logvar     
    
    @abstractmethod
    def reparametrize(self, mu, logvar):
        return None
    
    @abstractmethod
    def compute_kld(self, z_mu, z_logvar):
        return None
    
    @abstractmethod
    def compute_recon_loss(self, x, x_mu, x_logvar):
        return None
    
    @abstractmethod
    def get_loss(self, rec_loss, reg_loss):
        return None
    
    def call(self, x, training=True):
        z_mu, z_logvar = self.encode(x, training=training)
        z = self.reparametrize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z, training=training)
        return x_mu, x_logvar, z_mu, z_logvar
    
    def compute_negative_elbo(self, x, y, freebits=0.0, training=True):
        freebits = tf.cast(freebits, dtype=K.floatx())
        x_mu, x_logvar, z_mu, z_logvar = self(x, training=training)
        l_rec = self.compute_recon_loss(y, x_mu, x_logvar)
        l_kld = self.compute_kld(z_mu, z_logvar)
        l_kld = K.relu(l_kld - freebits*TF_LOG2) + freebits*TF_LOG2
        l_reg = K.sum(l_kld, 1)        
        return l_rec + l_reg, l_rec, l_reg
        
    def train_step(self, inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            _, l_rec, l_reg = self.compute_negative_elbo(
                x, y, freebits=0.05, training=True)
            loss, m_reg, m_rec = self.get_loss(l_rec, l_reg)
            self.elbo_loss.update_state(loss)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.custom_optimizer.apply_gradients(zip(gradients, self.trainable_variables))      
        self.reg_loss.update_state(m_reg)
        self.rec_loss.update_state(m_rec)
        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        x, y = data        
        _, l_rec, l_reg = self.compute_negative_elbo(
            x, y, freebits=0.05, training=False)
        log2 = tf.cast(K.log(2.0), dtype=K.floatx())
        loss, m_reg, m_rec = self.get_loss(l_rec, l_reg)
        self.elbo_loss.update_state(loss)
        self.reg_loss.update_state(m_reg)
        self.rec_loss.update_state(m_rec)
        return {m.name: m.result() for m in self.metrics} 
    
    
class GaussianBetaVAE(BaseVAE):
    
    def __init__(self, beta=1, *args, **kwargs):
        super(GaussianBetaVAE, self).__init__(*args, **kwargs)
        self.beta = tf.constant(beta, dtype=K.floatx(), name="beta")
            
    def reparametrize(self, mu, logvar):
        return tf_util.tf_gaussian_reparametrize(mu, logvar)
    
    def compute_kld(self, z_mu, z_logvar):
        return tf_util.tf_gaussian_kld(z_mu, z_logvar)
    
    def compute_recon_loss(self, x, x_mu, x_logvar):
        return -K.sum(tf_util.tf_gaussian_log_prob(x, x_mu, x_logvar), 1)
    
    def get_loss(self, rec_loss, reg_loss):
        reg_loss_scaled = self.beta*reg_loss
        loss_batch = reg_loss_scaled + rec_loss
        loss = K.mean(loss_batch) / TF_LOG2
        reg_metric = K.mean(reg_loss_scaled) / TF_LOG2
        rec_metric = K.mean(rec_loss) / TF_LOG2
        return loss, reg_metric, rec_metric
    
    
    
        