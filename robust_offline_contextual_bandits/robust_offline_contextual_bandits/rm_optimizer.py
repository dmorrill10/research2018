import tensorflow as tf
from robust_offline_contextual_bandits import optimizers


def rm(utility,
       ev,
       scale=1.0,
       non_negative=False,
       relax_simplex_constraint=False,
       regularization_bonus=None):
    allow_negative = not non_negative

    p = tf.nn.relu(utility - ev)
    if allow_negative: d = tf.nn.relu(-utility - ev)

    z = optimizers.sum_over_dims(p)
    if allow_negative: z += optimizers.sum_over_dims(d)
    if regularization_bonus is not None:
        z = z + regularization_bonus

    if relax_simplex_constraint:
        ev = tf.nn.relu(ev)
        p = tf.nn.relu(utility - ev)
        if allow_negative: d = tf.nn.relu(-utility - ev)

    delta = p
    if allow_negative: delta -= d

    c = tf.div_no_nan(tf.cast(scale, tf.float32), z)

    if z.shape[0].value == 1: z = optimizers.tile_to_dims(z, p.shape[0].value)
    default = (
        tf.zeros_like(z) if allow_negative or p.shape[0].value < 2
        else tf.fill(z.shape, 1.0 / p.shape[0].value)
    )  # yapf:disable
    return tf.where(tf.greater(z, 0), c * delta, default)


class RmBevL1VariableOptimizer(optimizers.StaticScaleMixin,
                               optimizers.RegretBasedVariableOptimizer):
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
        self._avoid_steps_beyond_max_gradient_norm = (
            avoid_steps_beyond_max_gradient_norm)
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
        avg_ev = self._get_or_make_slot(tf.zeros((1, self.num_columns())),
                                        'avg_ev')
        tf.summary.histogram('avg_ev', avg_ev)

        init = [init, avg_ev.initializer]
        if self._avoid_steps_beyond_max_gradient_norm:
            max_gradient_norm = self._get_or_make_slot(tf.zeros([]),
                                                       'max_gradient_norm')
            tf.summary.histogram('max_gradient_norm', max_gradient_norm)
            init.append(max_gradient_norm.initializer)
        return tf.group(*init)

    def _next_matrix_var(self,
                         next_regret_up,
                         next_regret_down,
                         scale=1.0,
                         regularization_bonus=None):
        p = tf.nn.relu(next_regret_up)
        allow_negative = not self.non_negative
        if allow_negative: d = tf.nn.relu(next_regret_down)

        z = optimizers.sum_over_dims(p)
        if allow_negative: z += optimizers.sum_over_dims(d)
        if regularization_bonus is not None:
            z = (z + regularization_bonus if self._additive_regularization else
                 tf.maximum(z, regularization_bonus))

        delta = p
        if allow_negative: delta -= d

        c = tf.div_no_nan(scale, z)

        if z.shape[0].value == 1:
            z = optimizers.tile_to_dims(z, p.shape[0].value)
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
        next_avg_ev = avg_ev.assign_add((iev - avg_ev) / t,
                                        use_locking=self._use_locking)

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

        regularization_bonus = self.updated_regularization_bonus(iregret_up,
                                                                 iregret_down,
                                                                 t=t)

        scale = self.scales()
        if self._avoid_steps_beyond_max_gradient_norm:
            max_gradient_norm = self.get_slot('max_gradient_norm')
            max_gradient_norm = max_gradient_norm.assign(
                tf.maximum(tf.reduce_sum(tf.abs(grad)), max_gradient_norm),
                use_locking=self._use_locking)
            scale = tf.minimum(scale, max_gradient_norm)
        next_var = self._var.assign(tf.reshape(
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
            regret_down.assign(next_regret_down,
                               use_locking=self._use_locking), next_avg_ev
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

        next_var = self._var.assign(tf.reshape(self.rm(utility, ev),
                                               self._var.shape),
                                    use_locking=self._use_locking)

        return tf.group(next_var, utility, ev)

    def rm(self, utility, ev, **kwargs):
        return rm(utility,
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
        return (tf.square(self.scales()) * (self._regularization_weight / t) *
                bonus)

    def dense_update(self, grad, num_updates=0):
        grad = self._with_fixed_dimensions(grad)
        t = tf.cast(num_updates + 1, tf.float32)
        iutility = self.utility(grad, scale=self.scales(), t=t)

        avg_ev = self.get_slot('avg_ev')
        iev = self.instantaneous_ev(iutility, scale=self.scales())

        iregret = [iutility - iev]
        allow_negative = not self.non_negative
        if allow_negative: iregret.append(-iutility - iev)

        avg_ev = avg_ev.assign_add((iev - avg_ev) / t,
                                   use_locking=self._use_locking)

        avg_utility = self.get_slot('avg_utility')
        avg_utility = avg_utility.assign_add((iutility - avg_utility) / t,
                                             use_locking=self._use_locking)

        regularization_bonus = self.updated_regularization_bonus(*iregret, t=t)

        next_var = self._var.assign(tf.reshape(
            self.rm(avg_utility,
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
            max_iregret = tf.maximum(max_iregret, tf.nn.relu(r))
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
        max_iregret = tf.nn.relu(iregret[0])
        for i in range(1, len(iregret)):
            max_iregret = tf.maximum(max_iregret, tf.nn.relu(iregret[i]))
        return avg_pos_regret.assign_add((max_iregret - avg_pos_regret) / t,
                                         use_locking=self._use_locking)


class RmSimMixin(RmMixin):
    @property
    def non_negative(self):
        return True


class RmL1VariableOptimizer(RmMixin, optimizers.StaticScaleMixin,
                            optimizers.GradEvBasedVariableOptimizer):
    pass


class RmSimVariableOptimizer(RmSimMixin, optimizers.StaticScaleMixin,
                             optimizers.GradEvBasedVariableOptimizer):
    pass


class RmInfVariableOptimizer(RmMixin, optimizers.StaticScaleMixin,
                             optimizers.GradEvBasedVariableOptimizer):
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


class RmL1AmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                _RmExtraRegularization, RmMixin,
                                optimizers.StaticScaleMixin,
                                optimizers.GradEvBasedVariableOptimizer):
    pass


class RmSimAmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                 _RmExtraRegularization, RmSimMixin,
                                 optimizers.StaticScaleMixin,
                                 optimizers.GradEvBasedVariableOptimizer):
    pass


class RmInfAmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                 _RmExtraRegularization, RmMixin,
                                 optimizers.StaticScaleMixin,
                                 optimizers.GradEvBasedVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnAmrrVariableOptimizer(_AvgMaxRegretRegularization,
                                _RmExtraRegularization, RmNnVariableOptimizer):
    pass


class RmL1AmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                 _RmExtraRegularization, RmMixin,
                                 optimizers.StaticScaleMixin,
                                 optimizers.GradEvBasedVariableOptimizer):
    pass


class RmSimAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                  _RmExtraRegularization, RmSimMixin,
                                  optimizers.StaticScaleMixin,
                                  optimizers.GradEvBasedVariableOptimizer):
    pass


class RmInfAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                  _RmExtraRegularization, RmMixin,
                                  optimizers.StaticScaleMixin,
                                  optimizers.GradEvBasedVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)


class RmNnAmarrVariableOptimizer(_AvgMaxAbsRegretRegularization,
                                 _RmExtraRegularization,
                                 RmNnVariableOptimizer):
    pass


class RmL1ArrVariableOptimizer(_AvgRegretRegularization,
                               _RmExtraRegularization, RmMixin,
                               optimizers.StaticScaleMixin,
                               optimizers.GradEvBasedVariableOptimizer):
    pass


class RmSimArrVariableOptimizer(_AvgRegretRegularization,
                                _RmExtraRegularization, RmSimMixin,
                                optimizers.StaticScaleMixin,
                                optimizers.GradEvBasedVariableOptimizer):
    pass


class RmInfArrVariableOptimizer(_AvgRegretRegularization,
                                _RmExtraRegularization, RmMixin,
                                optimizers.StaticScaleMixin,
                                optimizers.GradEvBasedVariableOptimizer):
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
