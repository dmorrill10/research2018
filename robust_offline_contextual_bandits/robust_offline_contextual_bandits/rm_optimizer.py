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

    c = tf.div_no_nan(scale, z)

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

    def _with_fixed_dimensions(self, v):
        return with_fixed_dimensions(
            v,
            independent_dimensions=self._independent_dimensions,
            dependent_columns=self._dependent_columns)

    @property
    def _matrix_var(self):
        return self._with_fixed_dimensions(self._var)

    def variables(self):
        return ([] if len(self._slots) < 1 else list(
            zip(*sorted(self._slots.items(), key=lambda e: e[0])))[-1])

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
            grad = tf.where(
                tf.greater(tf.abs(grad), self._clipvalue),
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
        clipvalue: float > 0. Cap on the size of gradient values. If None, defaults to infinite.
        amsgrad: bool. If True, uses the AMSGrad intermediate max step to ensure that the AdaGrad weights are non-decreasing.
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
            grad = tf.where(
                tf.greater(tf.abs(grad), self._clipvalue),
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

        return tf.group(
            self._var.assign_add(-lr * m_hat), next_m, next_v,
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
            grad = tf.where(
                tf.greater(tf.abs(grad), self._clipvalue),
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
        utility = self._get_or_make_slot(
            self._utility_initializer(self.shape), 'avg_utility')
        ev = self._get_or_make_slot(
            self._ev_initializer((1, self.num_columns())), 'avg_ev')

        tf.summary.histogram('avg_utility', utility)
        tf.summary.histogram('avg_ev', ev)
        return tf.group(utility.initializer, ev.initializer)

    def utility(self, grad, scale=1.0, descale=True, t=1):
        if self._clipvalue is not None:
            grad = tf.where(
                tf.greater(tf.abs(grad), self._clipvalue),
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
        return (self._scale * self.num_rows()
                if self._fractional_scale else self._scale)


class RmBevL1VariableOptimizer(StaticScaleMixin, RegretBasedVariableOptimizer):
    def __init__(self,
                 *args,
                 delay=True,
                 discount=0.0,
                 regularization_weight=None,
                 additive_regularization=False,
                 regularization_initializer=tf.zeros_initializer(),
                 avoid_steps_beyond_max_gradient_norm=False,
                 **kwargs):
        self._delay = delay
        self._discount = discount
        self._regularization_weight = regularization_weight
        self._additive_regularization = additive_regularization
        self._regularization_initializer = regularization_initializer
        self._avoid_steps_beyond_max_gradient_norm = avoid_steps_beyond_max_gradient_norm
        super().__init__(*args, **kwargs)

    def updated_regularization_bonus(self, *iregret, t=1):
        return None

    def scaled_regularization_bonus(self, bonus, t=1):
        if bonus is None: return None
        return (tf.square(self.scales()) * self._regularization_weight * bonus)

    @property
    def non_negative(self):
        return False

    def _create_slots(self):
        init = super()._create_slots()
        avg_ev = self._get_or_make_slot(
            tf.zeros((1, self.num_columns())), 'avg_ev')
        tf.summary.histogram('avg_ev', avg_ev)

        init = [init, avg_ev.initializer]
        if self._avoid_steps_beyond_max_gradient_norm:
            max_gradient_norm = self._get_or_make_slot(
                tf.zeros([]), 'max_gradient_norm')
            tf.summary.histogram('max_gradient_norm', max_gradient_norm)
            init.append(max_gradient_norm.initializer)
        return tf.group(*init)

    def _next_matrix_var(self,
                         next_regret_up,
                         next_regret_down,
                         scale=1.0,
                         regularization_bonus=None):
        p = plus(next_regret_up)
        allow_negative = not self.non_negative
        if allow_negative: d = plus(next_regret_down)

        z = sum_over_dims(p)
        if allow_negative: z += sum_over_dims(d)
        if regularization_bonus is not None:
            z = (z + regularization_bonus
                 if self._additive_regularization else tf.maximum(
                     z, regularization_bonus))

        delta = p
        if allow_negative: delta -= d

        c = tf.div_no_nan(scale, z)

        if z.shape[0].value == 1: z = tile_to_dims(z, p.shape[0].value)
        default = (
            tf.zeros_like(z) if allow_negative or p.shape[0].value < 2
            else tf.fill(z.shape, 1.0 / p.shape[0].value)
        )  # yapf:disable
        return tf.where(tf.greater(z, 0), c * delta, default)

    def dense_update(self, grad, num_updates=0):
        grad = self._with_fixed_dimensions(grad)
        utility = self.utility(grad)
        neg_utility = -utility
        iev = self.instantaneous_ev(utility, scale=self.scales(), descale=True)
        avg_ev = self.get_slot('avg_ev')

        t = tf.cast(num_updates + 1, tf.float32)
        next_avg_ev = avg_ev.assign_add(
            (iev - avg_ev) / t, use_locking=self._use_locking)

        regret_up = self.get_slot('cumulative_regret_up')
        regret_down = self.get_slot('cumulative_regret_down')

        iregret_up = utility - iev
        iregret_down = neg_utility - iev

        utility_is_gt_ev = tf.greater(utility, next_avg_ev)
        neg_utility_is_gt_ev = tf.greater(neg_utility, next_avg_ev)

        if self._delay:
            adj_regret_up = regret_up
            adj_regret_down = regret_down
        else:
            regret_up_is_neg = tf.less(regret_up, 0.0)
            adj_regret_up = tf.where(
                tf.logical_and(regret_up_is_neg, utility_is_gt_ev),
                self._discount * regret_up, regret_up)
            regret_down_is_neg = tf.less(regret_down, 0.0)
            adj_regret_down = tf.where(
                tf.logical_and(regret_down_is_neg, neg_utility_is_gt_ev),
                self._discount * regret_down, regret_down)

        next_regret_up = adj_regret_up + iregret_up
        next_regret_down = adj_regret_down + iregret_down

        if self._delay:
            regret_up_is_neg = tf.less(next_regret_up, 0.0)
            next_regret_up = tf.where(
                tf.logical_and(regret_up_is_neg, utility_is_gt_ev),
                self._discount * next_regret_up, next_regret_up)

            regret_down_is_neg = tf.less(next_regret_down, 0.0)
            next_regret_down = tf.where(
                tf.logical_and(regret_down_is_neg, neg_utility_is_gt_ev),
                self._discount * next_regret_down, next_regret_down)

        regularization_bonus = self.updated_regularization_bonus(
            iregret_up, iregret_down, t=t)

        scale = self.scales()
        if self._avoid_steps_beyond_max_gradient_norm:
            max_gradient_norm = self.get_slot('max_gradient_norm')
            max_gradient_norm = max_gradient_norm.assign(
                tf.maximum(tf.reduce_sum(tf.abs(grad)), max_gradient_norm),
                use_locking=self._use_locking)
            scale = tf.minimum(scale, max_gradient_norm)
        next_var = self._var.assign(
            tf.reshape(
                self._next_matrix_var(
                    next_regret_up,
                    next_regret_down,
                    scale=scale,
                    regularization_bonus=self.scaled_regularization_bonus(
                        regularization_bonus, t=t)), self._var.shape),
            use_locking=self._use_locking)

        updates = [
            next_var,
            regret_up.assign(next_regret_up, use_locking=self._use_locking),
            regret_down.assign(
                next_regret_down, use_locking=self._use_locking), next_avg_ev
        ]
        if regularization_bonus is not None:
            updates.append(regularization_bonus)
        if self._avoid_steps_beyond_max_gradient_norm:
            updates.append(max_gradient_norm)
        return tf.group(*updates)


class RmBevNnVariableOptimizer(RmBevL1VariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)

    @property
    def _matrix_var(self):
        return super()._matrix_var - self.scales()

    def _next_matrix_var(self, *args, **kwargs):
        return super()._next_matrix_var(*args, **kwargs) + self.scales()

    def scales(self):
        return super().scales() / 2.0


class RmBevInfVariableOptimizer(RmBevL1VariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmMixin(object):
    def __init__(self, *args, relax_simplex_constraint=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._relax_simplex_constraint = relax_simplex_constraint

    @property
    def non_negative(self):
        return False

    def dense_update(self, grad, num_updates=0):
        grad = self._with_fixed_dimensions(grad)
        t = tf.cast(num_updates + 1, tf.float32)
        utility = self.utility(grad, scale=self.scales(), t=t)

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


class _RmExtraRegularization(object):
    def __init__(self,
                 *args,
                 regularization_weight=1.0,
                 regularization_initializer=tf.zeros_initializer(),
                 **kwargs):
        self._regularization_weight = regularization_weight
        self._regularization_initializer = regularization_initializer
        super().__init__(*args, **kwargs)

    def updated_regularization_bonus(self, *iregret, t=1):
        raise NotImplementedError('Please override.')

    def scaled_regularization_bonus(self, bonus, t=1):
        if bonus is None: return None
        return (tf.square(self.scales()) *
                (self._regularization_weight / t) * bonus)

    def dense_update(self, grad, num_updates=0):
        grad = self._with_fixed_dimensions(grad)
        t = tf.cast(num_updates + 1, tf.float32)
        iutility = self.utility(grad, scale=self.scales(), t=t)

        avg_ev = self.get_slot('avg_ev')
        iev = self.instantaneous_ev(iutility, scale=self.scales())

        iregret = [iutility - iev]
        allow_negative = not self.non_negative
        if allow_negative: iregret.append(-iutility - iev)

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
                    regularization_bonus=self.scaled_regularization_bonus(
                        regularization_bonus, t=t)), self._var.shape),
            use_locking=self._use_locking)

        return tf.group(next_var, avg_utility, avg_ev, regularization_bonus)


class _AvgMaxRegretRegularization(object):
    def _create_slots(self):
        init = super()._create_slots()
        avg_max_pos_regret = self._get_or_make_slot(
            self._regularization_initializer(self.shape), 'avg_max_pos_regret')

        tf.summary.histogram('avg_max_pos_regret', avg_max_pos_regret)
        return tf.group(avg_max_pos_regret.initializer, init)

    def updated_regularization_bonus(self, *iregret, t=1):
        avg_max_pos_regret = self.get_slot('avg_max_pos_regret')
        max_iregret = avg_max_pos_regret
        for r in iregret:
            max_iregret = tf.maximum(max_iregret, plus(r))
        return avg_max_pos_regret.assign_add(
            (max_iregret - avg_max_pos_regret) / t,
            use_locking=self._use_locking)


class _AvgMaxAbsRegretRegularization(object):
    def _create_slots(self):
        init = super()._create_slots()
        avg_max_abs_regret = self._get_or_make_slot(
            self._regularization_initializer(self.shape), 'avg_max_abs_regret')

        tf.summary.histogram('avg_max_abs_regret', avg_max_abs_regret)
        return tf.group(avg_max_abs_regret.initializer, init)

    def updated_regularization_bonus(self, *iregret, t=1):
        avg_max_abs_regret = self.get_slot('avg_max_abs_regret')
        max_iregret = avg_max_abs_regret
        for r in iregret:
            max_iregret = tf.maximum(max_iregret, tf.abs(r))
        return avg_max_abs_regret.assign_add(
            (max_iregret - avg_max_abs_regret) / t,
            use_locking=self._use_locking)


class _AvgRegretRegularization(object):
    def _create_slots(self):
        init = super()._create_slots()
        avg_pos_regret = self._get_or_make_slot(
            self._regularization_initializer(self.shape), 'avg_pos_regret')

        tf.summary.histogram('avg_pos_regret', avg_pos_regret)
        return tf.group(avg_pos_regret.initializer, init)

    def updated_regularization_bonus(self, *iregret, t=1):
        avg_pos_regret = self.get_slot('avg_pos_regret')
        max_iregret = plus(iregret[0])
        for i in range(1, len(iregret)):
            max_iregret = tf.maximum(max_iregret, plus(iregret[i]))
        return avg_pos_regret.assign_add(
            (max_iregret - avg_pos_regret) / t, use_locking=self._use_locking)


class RmSimMixin(RmMixin):
    @property
    def non_negative(self):
        return True


class RmL1VariableOptimizer(RmMixin, StaticScaleMixin,
                            GradEvBasedVariableOptimizer):
    pass


class RmSimVariableOptimizer(RmSimMixin, StaticScaleMixin,
                             GradEvBasedVariableOptimizer):
    pass


class RmInfVariableOptimizer(RmMixin, StaticScaleMixin,
                             GradEvBasedVariableOptimizer):
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


class RmL1AmrrVariableOptimizer(
        _AvgMaxRegretRegularization, _RmExtraRegularization, RmMixin,
        StaticScaleMixin, GradEvBasedVariableOptimizer):
    pass


class RmSimAmrrVariableOptimizer(
        _AvgMaxRegretRegularization, _RmExtraRegularization, RmSimMixin,
        StaticScaleMixin, GradEvBasedVariableOptimizer):
    pass


class RmInfAmrrVariableOptimizer(
        _AvgMaxRegretRegularization, _RmExtraRegularization, RmMixin,
        StaticScaleMixin, GradEvBasedVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnAmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                _RmExtraRegularization, RmNnVariableOptimizer):
    pass


class RmL1AmarrVariableOptimizer(
        _AvgMaxAbsRegretRegularization, _RmExtraRegularization, RmMixin,
        StaticScaleMixin, GradEvBasedVariableOptimizer):
    pass


class RmSimAmarrVariableOptimizer(
        _AvgMaxAbsRegretRegularization, _RmExtraRegularization, RmSimMixin,
        StaticScaleMixin, GradEvBasedVariableOptimizer):
    pass


class RmInfAmarrVariableOptimizer(
        _AvgMaxAbsRegretRegularization, _RmExtraRegularization, RmMixin,
        StaticScaleMixin, GradEvBasedVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                 _RmExtraRegularization,
                                 RmNnVariableOptimizer):
    pass


class RmL1ArrVariableOptimizer(_AvgRegretRegularization,
                               _RmExtraRegularization, RmMixin,
                               StaticScaleMixin, GradEvBasedVariableOptimizer):
    pass


class RmSimArrVariableOptimizer(
        _AvgRegretRegularization, _RmExtraRegularization, RmSimMixin,
        StaticScaleMixin, GradEvBasedVariableOptimizer):
    pass


class RmInfArrVariableOptimizer(
        _AvgRegretRegularization, _RmExtraRegularization, RmMixin,
        StaticScaleMixin, GradEvBasedVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnArrVariableOptimizer(_AvgRegretRegularization,
                               _RmExtraRegularization, RmNnVariableOptimizer):
    pass


# RmBev with regularization bonus
class RmBevL1AmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                   RmBevL1VariableOptimizer):
    pass


class RmBevInfAmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                    RmBevInfVariableOptimizer):
    pass


class RmBevNnAmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                   RmBevNnVariableOptimizer):
    pass


class RmBevL1AmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                    RmBevL1VariableOptimizer):
    pass


class RmBevInfAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                     RmBevInfVariableOptimizer):
    pass


class RmBevNnAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                    RmBevNnVariableOptimizer):
    pass


class RmBevL1ArrVariableOptimizer(_AvgRegretRegularization,
                                  RmBevL1VariableOptimizer):
    pass


class RmBevInfArrVariableOptimizer(_AvgRegretRegularization,
                                   RmBevInfVariableOptimizer):
    pass


class RmBevNnArrVariableOptimizer(_AvgRegretRegularization,
                                  RmBevNnVariableOptimizer):
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
