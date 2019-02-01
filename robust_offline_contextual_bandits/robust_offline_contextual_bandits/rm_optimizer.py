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
       relax_simplex_constraint=False,
       regularization_bonus=None):
    allow_negative = not non_negative

    p = plus(utility - ev)
    if allow_negative: d = plus(-utility - ev)

    z = sum_over_dims(p)
    if allow_negative: z += sum_over_dims(d)
    if regularization_bonus is not None:
        z = z + regularization_bonus

    if relax_simplex_constraint:
        ev = plus(ev)
        p = plus(utility - ev)
        if allow_negative: d = plus(-utility - ev)

    delta = p
    if allow_negative: delta -= d

    c = scale / z

    if z.shape[0].value == 1: z = tile_to_dims(z, p.shape[0].value)
    default = (
        tf.zeros_like(z) if allow_negative or p.shape[0].value < 2
        else tf.fill(z.shape, 1.0 / p.shape[0].value)
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

    def _with_fixed_dimensions(self, v):
        return with_fixed_dimensions(
            v, independent_dimensions=self._independent_dimensions)

    @property
    def _matrix_var(self):
        return self._with_fixed_dimensions(self._var)

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
                 clipvalue=None,
                 **kwargs):
        self._utility_initializer = utility_initializer
        self._ev_initializer = ev_initializer
        self._momentum = momentum
        self._clipvalue = clipvalue
        super(GradEvBasedVariableOptimizer, self).__init__(*args, **kwargs)
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
        if not descale: m = m / scale
        if self._clipvalue is not None:
            grad = tf.where(
                tf.greater(tf.abs(grad), self._clipvalue),
                tf.sign(grad) * self._clipvalue, grad)
        return -grad + m

    def updated_utility(self, utility, scale=1.0, descale=True, t=1):
        cu = self.get_slot('avg_utility')
        if not descale: utility = scale * utility
        return cu.assign_add((utility - cu) / t, use_locking=self._use_locking)

    def updated_ev(self, utility, scale=1.0, descale=True, t=1):
        ev = self.get_slot('avg_ev')
        iev = tf.reduce_sum(self._matrix_var * utility, axis=0, keep_dims=True)
        if descale: iev = iev / scale
        return ev.assign_add((iev - ev) / t, use_locking=self._use_locking)


class StaticScaleVariableOptimizer(GradEvBasedVariableOptimizer):
    def __init__(self, *args, scale=1, **kwargs):
        self._scale = scale
        super(StaticScaleVariableOptimizer, self).__init__(*args, **kwargs)

    def scales(self):
        return self._scale


class RmMixin(object):
    def __init__(self, *args, relax_simplex_constraint=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._relax_simplex_constraint = relax_simplex_constraint

    @property
    def non_negative(self):
        return False

    def dense_update(self, grad, num_updates=0):
        grad = self._with_fixed_dimensions(grad)
        utility = self.utility(grad, scale=self.scales())

        t = tf.cast(num_updates + 1, tf.float32)
        ev = self.updated_ev(utility, scale=self.scales(), t=t)
        utility = self.updated_utility(utility, scale=self.scales(), t=t)

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
            non_negative=self.non_negative,
            **kwargs)


class _ExtraRegularization(object):
    def __init__(self, *args, regularization_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._regularization_weight = regularization_weight

    def updated_regularization_bonus(self, *iregret, t=1):
        raise NotImplementedError('Please override.')

    def dense_update(self, grad, num_updates=0):
        grad = self._with_fixed_dimensions(grad)
        iutility = self.utility(grad, scale=self.scales())

        avg_ev = self.get_slot('avg_ev')
        iev = tf.reduce_sum(
            self._matrix_var * iutility, axis=0, keep_dims=True)
        iev = iev / self.scales()

        iregret = [iutility - iev]
        allow_negative = not self.non_negative
        if allow_negative: iregret.append(grad - iev)

        t = tf.cast(num_updates + 1, tf.float32)
        avg_ev = avg_ev.assign_add(
            (iev - avg_ev) / t, use_locking=self._use_locking)

        avg_utility = self.get_slot('avg_utility')
        avg_utility = avg_utility.assign_add(
            (iutility - avg_utility) / t, use_locking=self._use_locking)

        regularization_bonus = self.updated_regularization_bonus(*iregret, t=t)

        next_var = self._var.assign(
            tf.reshape(
                self.rm(
                    avg_utility,
                    avg_ev,
                    regularization_bonus=regularization_bonus),
                self._var.shape),
            use_locking=self._use_locking)

        return tf.group(next_var, avg_utility, avg_ev, regularization_bonus)


class _AvgMaxRegretRegularization(_ExtraRegularization):
    def _create_slots(self):
        init = super()._create_slots()
        avg_max_pos_regret = self._get_or_make_slot(
            tf.zeros(self.shape), 'avg_max_pos_regret')

        tf.summary.histogram('avg_max_pos_regret', avg_max_pos_regret)
        return tf.group(avg_max_pos_regret.initializer, init)

    def updated_regularization_bonus(self, *iregret, t=1):
        avg_max_pos_regret = self.get_slot('avg_max_pos_regret')
        max_iregret = avg_max_pos_regret
        for r in iregret:
            max_iregret = tf.maximum(max_iregret, plus(r))
        avg_max_pos_regret = avg_max_pos_regret.assign_add(
            (max_iregret - avg_max_pos_regret) / t,
            use_locking=self._use_locking)
        scaled_avg_max_pos_regret = (tf.square(self.scales()) * (
            self._regularization_weight / t) * avg_max_pos_regret)
        return scaled_avg_max_pos_regret


class _AvgMaxAbsRegretRegularization(_ExtraRegularization):
    def _create_slots(self):
        init = super()._create_slots()
        avg_max_abs_regret = self._get_or_make_slot(
            tf.zeros(self.shape), 'avg_max_abs_regret')

        tf.summary.histogram('avg_max_abs_regret', avg_max_abs_regret)
        return tf.group(avg_max_abs_regret.initializer, init)

    def updated_regularization_bonus(self, *iregret, t=1):
        avg_max_abs_regret = self.get_slot('avg_max_abs_regret')
        max_iregret = avg_max_abs_regret
        for r in iregret:
            max_iregret = tf.maximum(max_iregret, tf.abs(r))
        avg_max_abs_regret = avg_max_abs_regret.assign_add(
            (max_iregret - avg_max_abs_regret) / t,
            use_locking=self._use_locking)
        scaled_avg_max_abs_regret = (tf.square(self.scales()) * (
            self._regularization_weight / t) * avg_max_abs_regret)
        return scaled_avg_max_abs_regret


class RmSimMixin(RmMixin):
    @property
    def non_negative(self):
        return True


class RmL1VariableOptimizer(RmMixin, StaticScaleVariableOptimizer):
    pass


class RmSimVariableOptimizer(RmSimMixin, StaticScaleVariableOptimizer):
    pass


class RmInfVariableOptimizer(RmMixin, StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnVariableOptimizer(RmInfVariableOptimizer):
    @property
    def _matrix_var(self):
        return super()._matrix_var - self.scales()

    def rm(self, *args, **kwargs):
        return super().rm(*args, **kwargs) + self.scales()

    def scales(self):
        return super().scales() / 2.0


class RmL1AmrrVariableOptimizer(_AvgMaxRegretRegularization, RmMixin,
                                StaticScaleVariableOptimizer):
    pass


class RmSimAmrrVariableOptimizer(_AvgMaxRegretRegularization, RmSimMixin,
                                 StaticScaleVariableOptimizer):
    pass


class RmInfAmrrVariableOptimizer(_AvgMaxRegretRegularization, RmMixin,
                                 StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnAmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                RmNnVariableOptimizer):
    pass


class RmL1AmarrVariableOptimizer(_AvgMaxAbsRegretRegularization, RmMixin,
                                 StaticScaleVariableOptimizer):
    pass


class RmSimAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization, RmSimMixin,
                                  StaticScaleVariableOptimizer):
    pass


class RmInfAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization, RmMixin,
                                  StaticScaleVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                 RmNnVariableOptimizer):
    pass


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
            return optimizer.dense_update(grad, self._num_updates)
