import tensorflow as tf
from tensorflow.keras import backend as K

# Set global parameters for this file
K.set_floatx('float64')

all = [
    "CoefficientOfDetermination"
]

def CoefficientOfDetermination(ret="fn"):
    if ret == "fn":
        return _coefficientOfDetermination_fn
    if ret == "cls":
        return _coefficientOfDetermination_cls()
    return None


def _coefficientOfDetermination_fn(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))   


class _coefficientOfDetermination_cls(tf.keras.metrics.Metric):
    
    def __init__(self, **kwargs):
        super(_coefficientOfDetermination_cls, self).__init__(name="R2", **kwargs)
        self.total_r2 = self.add_weight("total", initializer="zeros", dtype=K.floatx())
        self.count = self.add_weight("count", initializer="zeros", dtype=K.floatx())
        
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape, dtype=K.floatx()))
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        met = _coefficientOfDetermination_fn(y_true, y_pred)
        self.count.assign_add(1.)
        self.total_r2.assign_add(met)
        #return self.total_r2 / self.count
    
    def result(self):
        return self.total_r2 / self.count  