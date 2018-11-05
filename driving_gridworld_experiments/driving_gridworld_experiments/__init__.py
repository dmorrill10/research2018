import tensorflow as tf

from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.gridworld import DrivingGridworld
from driving_gridworld.rewards import SituationalReward

from tf_kofn_robust_policy_optimization.discounted_mdp import \
    state_successor_policy_evaluation_op, \
    dual_state_value_policy_evaluation_op, \
    state_distribution
from tf_kofn_robust_policy_optimization.robust.kofn import \
    KofnEvsAndWeights, \
    kofn_action_values

from research2018.tabular_cfr import TabularCfr
from research2018.kofn import KofnCfr


def new_road(headlight_range=3):
    return Road(
        headlight_range,
        Car(2, 0),
        obstacles=[
            Bump(-1, -1, prob_of_appearing=1 - 0.5**(1 / 4.0)),
            Pedestrian(-1, -1, speed=1, prob_of_appearing=1 - 0.5**(1 / 5.0))
        ],
        allowed_obstacle_appearance_columns=[{2}, {1}],
        allow_crashing=True)


def safety_info(root_probs,
                transitions,
                sa_safety_info,
                policy,
                discount=1.0,
                normalize=True):
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

    if normalize:
        state_safety_info = (1.0 - discount) * state_safety_info
    return tf.reduce_sum((root_probs * state_safety_info), axis=-1)


class UrdcKofnTabularCfr(KofnCfr):
    '''
    k-of-n mixin specific to tabular uncertain reward discounted continuing
    MDPs designed to override a `FixedParameterCfr` class.
    '''

    @classmethod
    def from_num_states_and_actions(cls,
                                    num_states,
                                    num_actions,
                                    cfr_cls=TabularCfr,
                                    **kwargs):
        return cls(cfr=cfr_cls.zeros(num_states, num_actions), **kwargs)

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


def tabular_road(headlight_range=3,
                 num_samples_per_cfr_iter=10,
                 n=100,
                 random_meta_mean=None,
                 use_slow_collision_as_offroad_base=True,
                 discount=0.99,
                 print_every=100):
    speed_limit = new_road(headlight_range=headlight_range).speed_limit()
    game = DrivingGridworld(lambda: new_road(headlight_range=headlight_range))
    num_reward_functions = n * num_samples_per_cfr_iter
    z = speed_limit * (
        speed_limit + int(use_slow_collision_as_offroad_base) + 1)

    if random_meta_mean is None:
        bcr = tf.fill([num_reward_functions], (1.0 - discount) / z)
        wc_ncer = -bcr
        cer = -tf.ones([num_reward_functions])
    else:
        bcr = tf.distributions.Exponential(random_meta_mean).sample(
            num_reward_functions)
        wc_ncer = -tf.distributions.Exponential(random_meta_mean).sample(
            num_reward_functions)
        cer = (z / (1.0 - discount)) * wc_ncer

    random_reward_function = SituationalReward(
        stopping_reward=tf.zeros([num_reward_functions]),
        wc_non_critical_error_reward=wc_ncer,
        bc_unobstructed_progress_reward=bcr,
        num_samples=num_reward_functions,
        critical_error_reward=cer,
        use_slow_collision_as_offroad_base=use_slow_collision_as_offroad_base)

    transitions, rfd_list, state_indices = game.road.tabulate(
        random_reward_function, print_every=print_every)

    transitions = tf.stack(transitions)
    reward_dataset = tf.stack(rfd_list)
    if random_meta_mean is not None:
        reward_dataset = (
            reward_dataset / -tf.reshape(cer, [1, 1, num_reward_functions]))

    root_probs = tf.one_hot(
        state_indices[game.road.copy().to_key()], depth=len(state_indices))
    return (root_probs, transitions, tf.transpose(reward_dataset, [2, 0, 1]),
            state_indices)
