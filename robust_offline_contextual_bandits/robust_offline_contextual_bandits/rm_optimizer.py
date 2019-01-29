from inspect import signature
import tensorflow as tf
import numpy as np
from tensorflow.python.training import optimizer
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


def plus(t):
    return tf.maximum(t, 0.0)


def sum_over_dims(t):
    return tf.reduce_sum(t, axis=0, keep_dims=True)


def tile_to_dims(t, num_dims):
    return tf.tile(t, [num_dims, 1])


def with_fixed_dimensions(t):
    if len(t.shape) < 2:
        return tf.expand_dims(t, axis=1)
    elif len(t.shape) > 2:
        d1 = np.prod([t.shape[i].value for i in range(len(t.shape) - 1)])
        return tf.reshape(t, [d1, t.shape[-1].value])
    else:
        return t


def rm(grad, ev, scale, non_negative=False):
    negative_grad = -grad

    dependent_directions = ev.shape[0].value < 2
    if dependent_directions: ev = tile_to_dims(ev, grad.shape[0].value)

    p = plus(negative_grad - ev)

    allow_negative = not non_negative
    if allow_negative: d = plus(grad - ev)

    if dependent_directions:
        z = sum_over_dims(p)
        if allow_negative: z += sum_over_dims(d)
        z = tile_to_dims(z, p.shape[0].value)
    else:
        z = p
        if allow_negative: z += d
    delta = p
    if allow_negative: delta -= d
    c = scale / z
    return tf.where(tf.greater(z, 0), c * delta, tf.zeros_like(z))


class VariableOptimizer(object):
    def __init__(self, var, use_locking=False, name=None):
        self._var = var
        self._matrix_var = with_fixed_dimensions(var)
        self.shape = tuple(v.value for v in self._matrix_var.shape)
        self._use_locking = use_locking
        self.name = type(self).__name__ if name is None else name
        self._slots = {}

    def variables(self):
        return self._slots.values()

    def num_rows(self):
        return self.shape[0]

    def num_columns(self):
        return self.shape[1]

    def _get_or_make_slot(self, val, name, **kwargs):
        with tf.variable_scope(self.name + '/' + name):
            self._slots[name] = ResourceVariable(
                val, trainable=False, **kwargs)
        return self._slots[name]

    def get_slot(self, name):
        return self._slots[name]

    def name_scope(self):
        return tf.name_scope(self.name + '/')


class GradEvBasedVariableOptimizer(VariableOptimizer):
    def __init__(self,
                 *args,
                 grad_initializer=tf.zeros_initializer(),
                 ev_initializer=tf.zeros_initializer(),
                 independent_directions=False,
                 **kwargs):
        super(GradEvBasedVariableOptimizer, self).__init__(*args, **kwargs)
        self._grad_initializer = grad_initializer
        self._ev_initializer = ev_initializer
        self._independent_directions = independent_directions
        self.initializer = self._create_slots()

    def _create_slots(self):
        with self.name_scope():
            grad = self._get_or_make_slot(
                self._grad_initializer(self.shape), 'cumulative_gradients')

            ev_shape = (self.shape if self._independent_directions else
                        (1, self.num_columns()))
            ev = self._get_or_make_slot(
                self._ev_initializer(ev_shape), 'cumulative_ev')

            tf.summary.histogram('cumulative_gradients', grad)
            tf.summary.histogram('cumulative_ev', ev)
        return tf.group(grad.initializer, ev.initializer)

    def updated_grad(self, grad, scale=1.0, descale=True):
        return self.get_slot('cumulative_gradients').assign_add(
            grad if descale else scale * grad, use_locking=self._use_locking)

    def updated_ev(self, grad, scale=1.0, descale=True):
        el = self._matrix_var * grad
        if not self._independent_directions:
            el = tf.reduce_sum(el, axis=0, keep_dims=True)
        if descale:
            inst_ev = el / -scale
        else:
            inst_ev = -el
        return self.get_slot('cumulative_ev').assign_add(
            inst_ev, use_locking=self._use_locking)


class StaticScaleVariableOptimizer(GradEvBasedVariableOptimizer):
    def __init__(self, *args, scale=1, **kwargs):
        super(StaticScaleVariableOptimizer, self).__init__(*args, **kwargs)
        self._scale = scale

    def scales(self):
        return self._scale


class RmVariableOptimizerMixin(object):
    def dense_update(self, grad):
        ev = self.updated_ev(grad, self.scales())
        grad = self.updated_grad(grad, self.scales())
        next_var = self._var.assign(
            tf.reshape(self.rm(grad, ev, self.scales()), self._var.shape),
            use_locking=self._use_locking)
        return tf.group(next_var, grad, ev)

    def sparse_update(self, grad):
        return self.dense_update(grad)

    def rm(self, grad, ev, scale):
        return rm(grad, ev, scale)


class RmSimVariableOptimizerMixin(RmVariableOptimizerMixin):
    def rm(self, grad, ev, scale):
        return rm(grad, ev, scale, non_negative=True)


class RmNnVariableOptimizerMixin(RmVariableOptimizerMixin):
    def rm(self, grad, ev, scale):
        return rm(grad, ev, scale, non_negative=True)


class RmL1VariableOptimizer(RmVariableOptimizerMixin,
                            StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_directions=False, **kwargs)


class RmInfVariableOptimizer(RmVariableOptimizerMixin,
                             StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_directions=True, **kwargs)


class RmSimVariableOptimizer(RmSimVariableOptimizerMixin,
                             StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_directions=False, **kwargs)


class RmNnVariableOptimizer(RmNnVariableOptimizerMixin,
                            StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_directions=True, **kwargs)


class CompositeVariableOptimizer(optimizer.Optimizer):
    @classmethod
    def combine(cls, *new_variable_optimizer, **kwargs):
        return cls(lambda var, i: new_variable_optimizer[i](var), **kwargs)

    def __init__(self,
                 new_variable_optimizer,
                 use_locking=False,
                 name=None,
                 var_list=[]):
        super(CompositeVariableOptimizer, self).__init__(
            use_locking,
            type(self).__name__ if name is None else name)
        self._new_opt = new_variable_optimizer
        self._optimizers = None
        self.initializer = None
        if len(var_list) > 0:
            self._create_slots(var_list)

    def variables(self):
        return sum(
            [list(opt.variables()) for opt in self._optimizers.values()], [])

    def _create_slots(self, var_list):
        if self._optimizers is None:
            self._optimizers = {}
            initializers = []
            pass_i = (len(
                signature(self._new_opt, follow_wrapped=False).parameters) > 1)
            with tf.variable_scope(self._name):
                for i in range(len(var_list)):
                    var = var_list[i]
                    self._optimizers[var] = (self._new_opt(var, i)
                                             if pass_i else self._new_opt(var))
                    initializers.append(self._optimizers[var].initializer)
                self.initializer = tf.group(*initializers)
        return self.initializer

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, var_list = zip(*grads_and_vars)
        if self._optimizers is None: self._create_slots(var_list)
        updates = []
        for i in range(len(grads)):
            updates.append(self._apply_dense(grads[i], var_list[i]))
        return tf.group(*updates)

    def _apply_dense(self, grad, var):
        with tf.variable_scope(self._name, reuse=True):
            return self._optimizers[var].dense_update(
                with_fixed_dimensions(grad))

    def _apply_sparse(self, grad, var):
        with tf.variable_scope(self._name, reuse=True):
            return self._optimizers[var].sparse_update(
                with_fixed_dimensions(grad))
