import tensorflow as tf
from robust_offline_contextual_bandits import optimizers


class MaxRegretRegularizedSdaMixin(object):
    def __init__(self,
                 *args,
                 min_reg_param=1e-15,
                 max_reg_param=1e15,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._min_reg_param = float(min_reg_param)
        self._max_reg_param = float(max_reg_param)

    @tf.function
    def _update_var(self, weights):
        return self._var.assign(weights, use_locking=self._use_locking)

    def dense_update(self, grad, num_updates=0):
        grad = self._with_fixed_dimensions(grad)

        t = tf.cast(num_updates + 1, tf.float32)
        utility = self.utility(grad, scale=self.scales(), t=t)

        ev = self.updated_ev(utility, scale=self.scales(), t=t, descale=False)
        utility = self.updated_utility(utility,
                                       scale=self.scales(),
                                       t=t,
                                       descale=True)

        next_var = self._update_var(
            tf.reshape(self.max_regret_regularized_sda(utility, ev, t),
                       self._var.shape))

        return tf.group(next_var, utility, ev)

    @tf.function
    def max_regret_regularized_sda(self, utility, ev, t):
        z = self.regret(utility, ev)
        prox_weight = optimizers.clip_by_value(z, self._min_reg_param,
                                               self._max_reg_param)

        # TODO: Not sure what the numerator should be here, but 1 seems way
        # too small. tf.square(self.scales()) should be a lot like RM.
        inverse_prox_weight = tf.math.divide_no_nan(tf.square(self.scales()),
                                                    prox_weight)

        weights = self.transform(inverse_prox_weight * utility)
        if z.shape[0] == 1: z = optimizers.tile_to_dims(z, weights.shape[0])

        return tf.where(tf.greater(z, 0.), weights,
                        self.transform(tf.zeros_like(z)))


class MaxRegretRegularizedSdaInfVariableOptimizer(
        MaxRegretRegularizedSdaMixin, optimizers.StaticScaleMixin,
        optimizers.GradEvBasedVariableOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, independent_dimensions=True, **kwargs)

    @tf.function
    def transform(self, weights):
        return optimizers.clip_by_value(weights, -self.scales(), self.scales())

    def regret(self, utility=None, ev=None):
        if utility is None:
            utility = self.avg_utility()
        if ev is None:
            ev = self.avg_ev()
        return self.scales() * optimizers.sum_over_dims(tf.abs(utility)) - ev


class MaxRegretRegularizedSdaNnVariableOptimizer(
        MaxRegretRegularizedSdaInfVariableOptimizer):
    @tf.function
    def transform(self, weights):
        return (
            optimizers.clip_by_value(weights, -self.scales(), self.scales()) +
            self.scales())

    @tf.function
    def scales(self):
        return super().scales() / 2.0

    def regret(self, utility=None, ev=None):
        if utility is None:
            utility = self.avg_utility()
        if ev is None:
            ev = self.avg_ev()
        return 2 * self.scales() * optimizers.sum_over_dims(
            tf.math.abs(utility)) - ev


class MaxRegretRegularizedSdaL2VariableOptimizer(
        MaxRegretRegularizedSdaMixin, optimizers.StaticScaleMixin,
        optimizers.GradEvBasedVariableOptimizer):
    @tf.function
    def transform(self, weights):
        squared_norm = optimizers.sum_over_dims(tf.square(weights))
        norm = tf.where(tf.greater(squared_norm, 0.0), tf.sqrt(squared_norm),
                        tf.zeros_like(squared_norm))
        return tf.where(tf.greater(norm, self.scales()),
                        weights * (self.scales() / norm), weights)

    def regret(self, utility=None, ev=None):
        if utility is None:
            utility = self.avg_utility()
        if ev is None:
            ev = self.avg_ev()
        squared_norm = optimizers.sum_over_dims(tf.square(utility))
        norm = tf.where(tf.greater(squared_norm, 0.0), tf.sqrt(squared_norm),
                        tf.zeros_like(squared_norm))
        return self.scales() * norm - ev
