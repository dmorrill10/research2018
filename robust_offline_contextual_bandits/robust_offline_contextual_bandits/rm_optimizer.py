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


def with_fixed_dimensions(t, independent_dimensions=False):
    if len(t.shape) < 2:
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


def rm(utility,
       ev,
       scale=1.0,
       non_negative=False,
       relax_simplex_constraint=False):
    if relax_simplex_constraint: ev = plus(ev)
    p = plus(utility - ev)

    allow_negative = not non_negative
    if allow_negative: d = plus(-utility - ev)

    z = sum_over_dims(p)
    if allow_negative: z += sum_over_dims(d)
    z = tile_to_dims(z, p.shape[0].value)

    delta = p
    if allow_negative: delta -= d
    c = scale / z
    default = (
        tf.zeros_like(z) if allow_negative
        else tf.fill(z.shape, 1.0 / max(p.shape[0].value, 2))
    )  # yapf:disable
    return tf.where(tf.greater(z, 0), c * delta, default)


class VariableOptimizer(object):
    def __init__(self,
                 var,
                 use_locking=False,
                 name=None,
                 independent_dimensions=False):
        self._var = var
        self._independent_dimensions = independent_dimensions
        self.shape = tuple(v.value for v in self._matrix_var.shape)
        self._use_locking = use_locking
        self.name = type(self).__name__ if name is None else name
        self._slots = {}

    @property
    def _matrix_var(self):
        return with_fixed_dimensions(
            self._var, independent_dimensions=self._independent_dimensions)

    def variables(self):
        return list(zip(*sorted(self._slots.items(), key=lambda e: e[0])))[-1]

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

    def sparse_update(self, grad, num_updates=0):
        return self.dense_update(grad, num_updates)


class GradEvBasedVariableOptimizer(VariableOptimizer):
    def __init__(self,
                 *args,
                 utility_initializer=tf.zeros_initializer(),
                 ev_initializer=tf.zeros_initializer(),
                 momentum=0.0,
                 **kwargs):
        super(GradEvBasedVariableOptimizer, self).__init__(*args, **kwargs)
        self._utility_initializer = utility_initializer
        self._ev_initializer = ev_initializer
        self._momentum = momentum
        with self.name_scope():
            self.initializer = self._create_slots()

    def _create_slots(self):
        utility = self._get_or_make_slot(
            self._utility_initializer(self.shape), 'avg_utility')
        ev = self._get_or_make_slot(
            self._ev_initializer((1, self.num_columns())), 'avg_ev')

        tf.summary.histogram('avg_utility', utility)
        tf.summary.histogram('avg_ev', ev)
        return tf.group(utility.initializer, ev.initializer)

    def utility(self, grad, scale=1.0, descale=True):
        m = self._momentum * self.get_slot('avg_utility')
        if not descale:
            m = m / scale
        return -grad + m

    def updated_utility(self, utility, scale=1.0, descale=True, num_updates=0):
        cu = self.get_slot('avg_utility')
        if not descale:
            utility = scale * utility
        return cu.assign_add(
            (utility - cu) / tf.cast(num_updates + 1, tf.float32),
            use_locking=self._use_locking)

    def updated_ev(self, utility, scale=1.0, descale=True, num_updates=0):
        ev = self.get_slot('avg_ev')
        iev = tf.reduce_sum(self._matrix_var * utility, axis=0, keep_dims=True)
        if descale:
            iev = iev / scale
        return ev.assign_add(
            (iev - ev) / tf.cast(num_updates + 1, tf.float32),
            use_locking=self._use_locking)


class StaticScaleVariableOptimizer(GradEvBasedVariableOptimizer):
    def __init__(self, *args, scale=1, **kwargs):
        super(StaticScaleVariableOptimizer, self).__init__(*args, **kwargs)
        self._scale = scale

    def scales(self):
        return self._scale


class RmMixin(object):
    def __init__(self, *args, relax_simplex_constraint=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._relax_simplex_constraint = (self.num_rows() < 2
                                          or relax_simplex_constraint)

    def dense_update(self, grad, num_updates=0):
        utility = self.utility(grad, scale=self.scales())
        ev = self.updated_ev(
            utility, scale=self.scales(), num_updates=num_updates)

        utility = self.updated_utility(
            utility, scale=self.scales(), num_updates=num_updates)
        next_var = self._var.assign(
            tf.reshape(self.rm(utility, ev), self._var.shape),
            use_locking=self._use_locking)

        return tf.group(next_var, utility, ev)

    def rm(self, utility, ev, **kwargs):
        return rm(
            utility,
            ev,
            scale=self.scales(),
            relax_simplex_constraint=self._relax_simplex_constraint,
            **kwargs)


class RmSimMixin(RmMixin):
    def rm(self, *args):
        return super().rm(*args, non_negative=True)


class RmL1VariableOptimizer(RmMixin, StaticScaleVariableOptimizer):
    pass


class RmSimVariableOptimizer(RmSimMixin, StaticScaleVariableOptimizer):
    pass


class RmInfVariableOptimizer(RmMixin, StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnVariableOptimizer(RmSimMixin, StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class CompositeOptimizer(optimizer.Optimizer):
    @classmethod
    def combine(cls, *new_variable_optimizer, **kwargs):
        return cls(lambda var, i: new_variable_optimizer[i](var), **kwargs)

    def __init__(self,
                 new_variable_optimizer,
                 use_locking=False,
                 name=None,
                 var_list=[]):
        super(CompositeOptimizer, self).__init__(use_locking,
                                                 type(self).__name__
                                                 if name is None else name)
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
            updates.append(
                self._apply_gradients(self._optimizers[i], grads[i]))
        updates.append(
            self._num_updates.assign_add(1, use_locking=self._use_locking))
        return tf.group(*updates)

    def _apply_gradients(self, optimizer, grad):
        with tf.variable_scope(self._name, reuse=True):
            return optimizer.dense_update(
                with_fixed_dimensions(grad), self._num_updates)
