import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np


def new_t_inv_gd_optimizer(learning_rate):
    return tf.train.GradientDescentOptimizer(
        learning_rate=tf.train.inverse_time_decay(
            learning_rate, tf.train.get_global_step(), 1, 1))


def sum_over_dims(t):
    return tf.reduce_sum(t, axis=0, keep_dims=True)


def tile_to_dims(t, num_dims):
    return tf.tile(t, [num_dims, 1])


def with_fixed_dimensions(t,
                          independent_dimensions=False,
                          dependent_columns=False):
    if dependent_columns:
        return tf.reshape(t, [tf.size(t), 1])
    elif len(t.shape) < 2:
        return tf.expand_dims(t, axis=0)
    else:
        num_columns = t.shape[-1].value
        num_dimensions = (
            np.prod([t.shape[i].value for i in range(len(t.shape) - 1)])
            if len(t.shape) > 2 else
            t.shape[0].value
        )  # yapf:disable
        if independent_dimensions:
            num_columns = num_dimensions * num_columns
            num_dimensions = 1
        return tf.reshape(t, [num_dimensions, num_columns])


class VariableOptimizer(object):
    def __init__(self,
                 var,
                 use_locking=False,
                 name=None,
                 independent_dimensions=False,
                 dependent_columns=False):
        assert not (independent_dimensions and dependent_columns)
        self._var = var
        self._independent_dimensions = independent_dimensions
        self._dependent_columns = dependent_columns
        self.shape = tuple(v.value for v in self._matrix_var.shape)
        self._use_locking = use_locking
        self.name = type(self).__name__ if name is None else name
        self._slots = {}

    @property
    def _matrix_var(self):
        return self._with_fixed_dimensions(self._var)

    def _with_fixed_dimensions(self, v):
        return with_fixed_dimensions(
            v,
            independent_dimensions=self._independent_dimensions,
            dependent_columns=self._dependent_columns)

    def variables(self):
        return ([] if len(self._slots) < 1 else list(
            zip(*sorted(self._slots.items(), key=lambda e: e[0])))[-1])

    def num_rows(self):
        return self.shape[0]

    def num_columns(self):
        return self.shape[1]

    def _get_or_make_slot(self, val, name, **kwargs):
        with tf.variable_scope(self.name + '/' + name):
            self._slots[name] = ResourceVariable(val,
                                                 trainable=False,
                                                 **kwargs)
        return self._slots[name]

    def get_slot(self, name):
        return self._slots[name]

    def name_scope(self):
        return tf.name_scope(self.name + '/')

    def sparse_update(self, grad, num_updates=0):
        return self.dense_update(grad, num_updates)

    def instantaneous_ev(self, utility, scale=1.0, descale=True):
        iev = tf.reduce_sum(self._matrix_var * utility, axis=0, keep_dims=True)
        if descale: iev = iev / scale
        return iev


class GradientDescentVariableOptimizer(VariableOptimizer):
    def __init__(self,
                 *args,
                 step_size=0.1,
                 clipvalue=None,
                 linear_step_size_decrease=False,
                 sqrt_step_size_decrease=False,
                 **kwargs):
        self._clipvalue = clipvalue
        self._step_size = step_size
        self._linear_step_size_decrease = linear_step_size_decrease
        self._sqrt_step_size_decrease = sqrt_step_size_decrease
        self.initializer = tf.group()
        super(GradientDescentVariableOptimizer, self).__init__(*args, **kwargs)

    def dense_update(self, grad, num_updates=0):
        if self._clipvalue is not None:
            grad = tf.where(tf.greater(tf.abs(grad), self._clipvalue),
                            tf.sign(grad) * self._clipvalue, grad)
        ss = self._step_size
        if self._linear_step_size_decrease:
            t = tf.cast(num_updates + 1, tf.float32)
            ss = ss / t
        elif self._sqrt_step_size_decrease:
            t = tf.cast(num_updates + 1, tf.float32)
            ss = ss / tf.sqrt(t)
        return self._var.assign_add(-self._step_size * grad)


class AdamVariableOptimizer(VariableOptimizer):
    '''Adam optimizer.'''

    def __init__(self,
                 *args,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 clipvalue=None,
                 amsgrad=True,
                 init_with_first_grad=False,
                 debias=True,
                 **kwargs):
        '''
        Default parameters follow those provided in the original paper.

        Arguments:
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Momentum weight. Generally close to 1.
        beta_2: float, 0 < beta < 1. Adagrad weight. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
        clipvalue: float > 0. Cap on the size of gradient values. If None,
            defaults to infinite.
        amsgrad: bool. If True, uses the AMSGrad intermediate max step to
            ensure that the AdaGrad weights are non-decreasing.
        '''
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = tf.keras.backend.epsilon(
        ) if epsilon is None else epsilon
        self._clipvalue = clipvalue
        self._amsgrad = amsgrad
        self._init_with_first_grad = init_with_first_grad
        self._debias = debias
        super(AdamVariableOptimizer, self).__init__(*args, **kwargs)
        with self.name_scope():
            self.initializer = self._create_slots()

    def _create_slots(self):
        v = self._get_or_make_slot(tf.zeros(self._var.shape), 'v')
        m = self._get_or_make_slot(tf.zeros(self._var.shape), 'm')

        tf.summary.histogram('v', v)
        tf.summary.histogram('m', m)

        init = [v.initializer, m.initializer]

        if self._amsgrad:
            v_hat = self._get_or_make_slot(tf.zeros(self._var.shape), 'v_hat')
            tf.summary.histogram('v_hat', v_hat)
            init.append(v_hat.initializer)
        return tf.group(*init)

    def dense_update(self, grad, num_updates=0):
        if self._clipvalue is not None:
            grad = tf.where(tf.greater(tf.abs(grad), self._clipvalue),
                            tf.sign(grad) * self._clipvalue, grad)
        m = self.get_slot('m')
        v = self.get_slot('v')

        if self._init_with_first_grad and num_updates == 0:
            next_m = grad
            next_v = tf.square(grad)
        else:
            next_m = self._beta_1 * m + (1.0 - self._beta_1) * grad
            next_v = self._beta_2 * v + (1.0 - self._beta_2) * tf.square(grad)

        next_m = m.assign(next_m, use_locking=self._use_locking)
        next_v = v.assign(next_v, use_locking=self._use_locking)

        if self._debias:
            t = tf.cast(num_updates + 1, tf.float32)
            m_hat = next_m / (1.0 - tf.pow(self._beta_1, t))
            v_hat = next_v / (1.0 - tf.pow(self._beta_2, t))
        else:
            m_hat = next_m
            v_hat = next_v

        optional_updates = []
        if self._amsgrad:
            v_hat_prev = self.get_slot('v_hat')
            v_hat = v_hat_prev.assign(tf.maximum(v_hat_prev, v_hat))
            optional_updates.append(v_hat)
        lr = self._lr / (tf.sqrt(v_hat) + self._epsilon)

        return tf.group(self._var.assign_add(-lr * m_hat), next_m, next_v,
                        *optional_updates)
