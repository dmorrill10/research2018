import tensorflow as tf
from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Bump, Pedestrian
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    state_successor_policy_evaluation_op, \
    dual_state_value_policy_evaluation_op, \
    state_distribution
from tf_kofn_robust_policy_optimization.robust.kofn import \
    KofnEvsAndWeights, \
    kofn_action_values
from research2018.tabular_cfr import TabularCfr
from research2018.kofn import KofnCfr


def new_road(headlight_range=3,
             allow_crashing=False,
             car=None,
             obstacles=None,
             allowed_obstacle_appearance_columns=None):
    return Road(
        headlight_range,
        Car(2, 0) if car is None else car,
        obstacles=([
            Bump(-1, -1, prob_of_appearing=0.5),
            Pedestrian(-1, -1, speed=1, prob_of_appearing=0.5)
        ] if obstacles is None else obstacles),
        allowed_obstacle_appearance_columns=(
            [{2}, {1}] if allowed_obstacle_appearance_columns is None else
            allowed_obstacle_appearance_columns),
        allow_crashing=allow_crashing)


def safety_info(root_probs, transitions, sa_safety_info, policy, discount=1.0):
    '''Assumes the first dimension is a batch dimension.'''
    state_safety_info = dual_state_value_policy_evaluation_op(
        transitions, policy, sa_safety_info, gamma=discount)

    if len(state_safety_info.shape) < 2:
        state_safety_info = tf.expand_dims(state_safety_info, 0)
    else:
        state_safety_info = tf.transpose(state_safety_info)

    root_probs = tf.convert_to_tensor(root_probs)
    if len(root_probs.shape) < 2:
        root_probs = tf.expand_dims(root_probs, 0)
    return tf.reduce_sum(root_probs * state_safety_info, axis=-1)


class UrdcKofnTabularCfr(KofnCfr):
    '''
    k-of-n mixin specific to tabular uncertain reward discounted continuing
    MDPs designed to override a `FixedParameterCfr` class.
    '''

    @classmethod
    def from_num_states_and_actions(cls, num_states, num_actions, **kwargs):
        return cls(cfr=TabularCfr.zeros(num_states, num_actions), **kwargs)

    @classmethod
    def train_env(cls,
                  root_probs,
                  transitions,
                  reward_dataset,
                  discount,
                  n,
                  kofn_opponent,
                  num_samples=1):
        def env(policy):
            '''world X state'''
            v = dual_state_value_policy_evaluation_op(
                transitions, policy, reward_dataset, gamma=discount)
            '''state X action X world'''
            q = tf.transpose(
                reward_dataset +
                discount * tf.tensordot(v, transitions, axes=[-1, -1]),
                [1, 2, 0])
            v = tf.transpose(v)

            offset = 0
            kofn_q = []
            for sample_idx in range(num_samples):
                next_offset = n * (sample_idx + 1)
                sample_v = v[:, offset:next_offset]
                sample_q = q[:, :, offset:next_offset]

                kofn_evs_and_weights = KofnEvsAndWeights(
                    sample_v, kofn_opponent, context_weights=root_probs)

                kofn_q.append(
                    kofn_action_values(sample_q,
                                       kofn_evs_and_weights.world_weights))

                offset = next_offset
            return tf.reduce_mean(tf.stack(kofn_q, -1), axis=-1)

        return env

    @classmethod
    def test_env(cls,
                 root_probs,
                 transitions,
                 reward_dataset,
                 discount,
                 n,
                 kofn_opponent,
                 num_samples=1):
        def env(policy):
            '''world X state'''
            v = dual_state_value_policy_evaluation_op(
                transitions, policy, reward_dataset, gamma=discount)
            v = tf.transpose(v)
            offset = 0
            kofn_ev = []
            for sample_idx in range(num_samples):
                next_offset = n * (sample_idx + 1)

                kofn_ev.append(
                    KofnEvsAndWeights(
                        v[:, offset:next_offset],
                        kofn_opponent,
                        context_weights=root_probs).ev)

                offset = next_offset
            return tf.reduce_mean(tf.stack(kofn_ev, -1), axis=-1)

        return env

    @property
    def policy(self):
        return self.cfr.policy

    def state_successor_rep(self, transitions, discount=1.0):
        transitions = tf.convert_to_tensor(transitions)
        return state_successor_policy_evaluation_op(
            transitions, self.policy(), gamma=discount)

    def state_distribution(self, root_probs, transitions, **kwargs):
        return state_distribution(
            self.state_successor_rep(transitions, **kwargs), root_probs)

    def state_action_distribution(self, **kwargs):
        return (tf.expand_dims(self.state_distribution(**kwargs), axis=-1) *
                self.policy())
