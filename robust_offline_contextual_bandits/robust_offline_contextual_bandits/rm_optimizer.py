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


def rm(grad, ev, scale):
    negative_grad = -grad

    independent_directions = ev.shape[0].value < 2
    if independent_directions:
        ev = tile_to_dims(ev, grad.shape[0].value)

    p = plus(negative_grad - ev)
    d = plus(grad - ev)

    if independent_directions:
        z = p + d
    else:
        sum_p = sum_over_dims(p)
        sum_d = sum_over_dims(d)
        z = sum_p + sum_d
        z = tile_to_dims(z, p.shape[0].value)
    c = scale / z
    return tf.where(tf.greater(z, 0), c * (p - d), tf.zeros_like(z))


class VariableOptimizer(object):
    def __init__(self, var, use_locking=False, name=None):
        self._var = var
        self._matrix_var = with_fixed_dimensions(var)
        self.shape = [v.value for v in self._matrix_var.shape]
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
                 grad_initializer=tf.zeros_initializer,
                 ev_initializer=tf.zeros_initializer,
                 independent_directions=False,
                 **kwargs):
        super(GradEvBasedVariableOptimizer, self).__init__(*args, **kwargs)
        self._grad_initializer = grad_initializer
        self._ev_initializer = ev_initializer
        self._independent_directions = independent_directions

    def create_slots(self):
        with self.name_scope():
            grad = self._get_or_make_slot(self._grad_initializer()(self.shape),
                                          'cumulative_gradients')

            if self._independent_directions:
                ev = self._get_or_make_slot(self._ev_initializer()(self.shape),
                                            'cumulative_ev')
            else:
                ev = self._get_or_make_slot(self._ev_initializer()(
                    [1, self.num_columns()]), 'cumulative_ev')

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
            tf.reshape(rm(grad, ev, self.scales()), self._var.shape),
            use_locking=self._use_locking)
        return tf.group(next_var, grad, ev)

    def sparse_update(self, grad):
        return self.dense_update(grad)


class RmVariableOptimizer(RmVariableOptimizerMixin,
                          StaticScaleVariableOptimizer):
    pass


class CompositeVariableOptimizer(optimizer.Optimizer):
    def __init__(self,
                 variable_optimizer_factory,
                 use_locking=False,
                 name=None):
        super(CompositeVariableOptimizer, self).__init__(
            use_locking,
            type(self).__name__ if name is None else name)
        self._variable_optimizer_factory = variable_optimizer_factory
        self._variable_optimizers = None

    def variables(self):
        return sum([
            list(opt.variables())
            for opt in self._variable_optimizers.values()
        ], [])

    def _create_slots(self, var_list):
        self._variable_optimizers = {}
        initializers = []
        with tf.variable_scope(self._name):
            for i in range(len(var_list)):
                var = var_list[i]
                self._variable_optimizers[
                    var] = self._variable_optimizer_factory(var, i)
                initializers.append(
                    self._variable_optimizers[var].create_slots())
        return tf.group(*initializers)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, var_list = zip(*grads_and_vars)
        if self._variable_optimizers is None:
            self._create_slots(var_list)
        updates = []
        for i in range(len(grads)):
            updates.append(self._apply_dense(grads[i], var_list[i]))
        return tf.group(*updates)

    def _apply_dense(self, grad, var):
        with tf.variable_scope(self._name, reuse=True):
            return self._variable_optimizers[var].dense_update(
                with_fixed_dimensions(grad))

    def _apply_sparse(self, grad, var):
        with tf.variable_scope(self._name, reuse=True):
            return self._variable_optimizers[var].sparse_update(
                with_fixed_dimensions(grad))


class RmOptimizer(CompositeVariableOptimizer):
    def __init__(self,
                 polytope_scales=[],
                 independent_directions=[],
                 grad_initializer=tf.zeros_initializer,
                 ev_initializer=tf.zeros_initializer,
                 **kwargs):
        def f(var, i):
            scale = polytope_scales[i]
            id = (independent_directions[i]
                  if len(independent_directions) > i else False)
            return RmVariableOptimizer(
                var,
                scale=scale,
                independent_directions=id,
                grad_initializer=grad_initializer,
                ev_initializer=ev_initializer,
                name='RM_layer_{}_s{}'.format(i, scale))

        super(RmOptimizer, self).__init__(f, **kwargs)
