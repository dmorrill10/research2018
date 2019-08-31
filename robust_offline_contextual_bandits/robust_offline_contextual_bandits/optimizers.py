from inspect import signature
import tensorflow as tf
from tensorflow.python.training import optimizer
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np


def new_t_inv_gd_optimizer(learning_rate):
    return tf.train.GradientDescentOptimizer(
        learning_rate=tf.train.inverse_time_decay(
            learning_rate, tf.train.get_global_step(), 1, 1))


def sum_over_dims(t):
    return tf.reduce_sum(t, axis=0, keep_dims=True)


def max_over_dims(t):
    return tf.reduce_max(t, axis=0, keep_dims=True)


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


class RegretBasedVariableOptimizer(VariableOptimizer):
    def __init__(self,
                 *args,
                 initializer=tf.zeros_initializer(),
                 clipvalue=None,
                 **kwargs):
        self._initializer = initializer
        self._clipvalue = clipvalue
        super(RegretBasedVariableOptimizer, self).__init__(*args, **kwargs)
        with self.name_scope():
            self.initializer = self._create_slots()

    def _create_slots(self):
        cumulative_regret_up = self._get_or_make_slot(
            self._initializer(self.shape), 'cumulative_regret_up')
        cumulative_regret_down = self._get_or_make_slot(
            self._initializer(self.shape), 'cumulative_regret_down')

        tf.summary.histogram('cumulative_regret_up', cumulative_regret_up)
        tf.summary.histogram('cumulative_regret_down', cumulative_regret_down)
        return tf.group(cumulative_regret_up.initializer,
                        cumulative_regret_down.initializer)

    def utility(self, grad):
        if self._clipvalue is not None:
            grad = tf.where(tf.greater(tf.abs(grad), self._clipvalue),
                            tf.sign(grad) * self._clipvalue, grad)
        return -grad

    def dense_update(self, grad, num_updates=0):
        raise NotImplementedError('Please override.')


class GradEvBasedVariableOptimizer(VariableOptimizer):
    def __init__(self,
                 *args,
                 utility_initializer=tf.zeros_initializer(),
                 ev_initializer=tf.zeros_initializer(),
                 momentum=0.0,
                 clipvalue=None,
                 use_linear_weight=False,
                 **kwargs):
        self._utility_initializer = utility_initializer
        self._ev_initializer = ev_initializer
        self._momentum = momentum
        self._clipvalue = clipvalue
        self._use_linear_weight = use_linear_weight
        super(GradEvBasedVariableOptimizer, self).__init__(*args, **kwargs)
        with self.name_scope():
            self.initializer = self._create_slots()

    def _create_slots(self):
        utility = self._get_or_make_slot(self._utility_initializer(self.shape),
                                         'avg_utility')
        ev = self._get_or_make_slot(
            self._ev_initializer((1, self.num_columns())), 'avg_ev')

        tf.summary.histogram('avg_utility', utility)
        tf.summary.histogram('avg_ev', ev)
        return tf.group(utility.initializer, ev.initializer)

    def utility(self, grad, scale=1.0, descale=True, t=1):
        if self._clipvalue is not None:
            grad = tf.where(tf.greater(tf.abs(grad), self._clipvalue),
                            tf.sign(grad) * self._clipvalue, grad)
        if self._momentum is not None and self._momentum > 0:
            m = self._momentum * self.get_slot('avg_utility')
            if not descale: m = m / scale
            u = m - (1.0 - self._momentum) * grad
        else:
            u = -grad
        if self._use_linear_weight:
            u = t * u
        return u

    def updated_utility(self, utility, scale=1.0, descale=True, t=1):
        cu = self.get_slot('avg_utility')
        if not descale: utility = scale * utility
        return cu.assign_add((utility - cu) / t, use_locking=self._use_locking)

    def updated_ev(self, utility, scale=1.0, descale=True, t=1):
        ev = self.get_slot('avg_ev')
        iev = self.instantaneous_ev(utility, scale=scale, descale=descale)
        return ev.assign_add((iev - ev) / t, use_locking=self._use_locking)


class StaticScaleMixin(object):
    def __init__(self, *args, scale=1, fractional_scale=False, **kwargs):
        self._scale = scale
        self._fractional_scale = fractional_scale
        super(StaticScaleMixin, self).__init__(*args, **kwargs)

    def scales(self):
        return (self._scale *
                self.num_rows() if self._fractional_scale else self._scale)


class MaxRegretRegularizedSdaMixin(object):
    def __init__(self, *args, min_reg_param=1e-5, max_reg_param=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_reg_param = float(min_reg_param)
        self._max_reg_param = float(max_reg_param)

    def dense_update(self, grad, num_updates=0):
        grad = self._with_fixed_dimensions(grad)
        t = tf.cast(num_updates + 1, tf.float32)
        utility = self.utility(grad, scale=self.scales(), t=t)

        ev = self.updated_ev(utility, scale=self.scales(), t=t)
        utility = self.updated_utility(utility, scale=self.scales(), t=t)

        next_var = self._var.assign(tf.reshape(
            self.max_regret_regularized_sda(utility, ev, t), self._var.shape),
                                    use_locking=self._use_locking)

        return tf.group(next_var, utility, ev)

    def max_regret_regularized_sda(self, utility, ev, t):
        p = tf.nn.relu(utility - ev)
        d = tf.nn.relu(-utility - ev)

        z = tf.maximum(max_over_dims(p), max_over_dims(d))

        inverse_prox_weight = tf.minimum(self._max_reg_param,
                                         tf.maximum(self._min_reg_param, z))

        final_prox_weight = tf.math.div_no_nan(tf.math.sqrt(t), inverse_prox_weight)

        if z.shape[0].value == 1: z = tile_to_dims(z, p.shape[0].value)

        weights = tf.maximum(
            -self.scales(),
            tf.minimum(self.scales(), final_prox_weight * utility))

        return tf.where(tf.greater(z, 0), weights, tf.zeros_like(z))


class MaxRegretRegularizedSdaInfVariableOptimizer(MaxRegretRegularizedSdaMixin,
                                                  StaticScaleMixin,
                                                  GradEvBasedVariableOptimizer
                                                  ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class MaxRegretRegularizedSdaNnVariableOptimizer(
        MaxRegretRegularizedSdaInfVariableOptimizer):
    @property
    def _matrix_var(self):
        return super()._matrix_var - self.scales()

    def max_regret_regularized_sda(self, *args, **kwargs):
        return (super().max_regret_regularized_sda(*args, **kwargs) +
                self.scales())

    def scales(self):
        return super().scales() / 2.0


class CompositeOptimizer(optimizer.Optimizer):
    @classmethod
    def combine(cls, *new_variable_optimizer, **kwargs):
        return cls(lambda var, i: new_variable_optimizer[i](var), **kwargs)

    def __init__(self,
                 new_variable_optimizer,
                 use_locking=False,
                 name=None,
                 var_list=[]):
        super(CompositeOptimizer,
              self).__init__(use_locking,
                             type(self).__name__ if name is None else name)
        self._new_opt = new_variable_optimizer
        self._optimizers = None
        self.initializer = None
        with tf.variable_scope(self._name, reuse=True):
            self._num_updates = ResourceVariable(0, name='num_updates')
        if len(var_list) > 0:
            self._create_slots(var_list)

    def variables(self):
        return sum([list(opt.variables()) for opt in self._optimizers],
                   [self._num_updates])

    def _create_slots(self, var_list):
        if self._optimizers is None:
            self._optimizers = []
            initializers = [self._num_updates.initializer]
            pass_i = (len(
                signature(self._new_opt, follow_wrapped=False).parameters) > 1)
            with tf.variable_scope(self._name):
                for i in range(len(var_list)):
                    var = var_list[i]
                    self._optimizers.append((self._new_opt(var, i) if pass_i
                                             else self._new_opt(var)))
                    initializers.append(self._optimizers[-1].initializer)
                self.initializer = tf.group(*initializers)
        return self.initializer

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, var_list = zip(*grads_and_vars)
        if self._optimizers is None: self._create_slots(var_list)
        updates = []
        for i in range(len(grads)):
            updates.append(self._apply_gradients(self._optimizers[i],
                                                 grads[i]))
        updates.append(
            self._num_updates.assign_add(1, use_locking=self._use_locking))
        return tf.group(*updates)

    def _apply_gradients(self, optimizer, grad):
        with tf.variable_scope(self._name, reuse=True):
            return optimizer.dense_update(grad, self._num_updates)
