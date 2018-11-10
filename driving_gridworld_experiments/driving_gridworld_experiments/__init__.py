import tensorflow as tf

from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.gridworld import DrivingGridworld
from driving_gridworld.rewards import \
    DebrisPerceptionReward, \
    fixed_ditch_bonus, \
    critical_reward_for_fixed_ditch_bonus

from tf_kofn_robust_policy_optimization.discounted_mdp import \
    state_successor_policy_evaluation_op, \
    dual_state_value_policy_evaluation_op, \
    state_distribution
from tf_kofn_robust_policy_optimization.robust.kofn import \
    KofnEvsAndWeights, \
    kofn_action_values

from research2018.tabular_cfr import TabularCfr
from research2018.kofn import KofnCfr


def new_road(headlight_range=2):
    return Road(
        headlight_range,
        Car(2, 0),
        obstacles=[
            Bump(-1, -1, prob_of_appearing=0.1),
            Pedestrian(-1, -1, speed=1, prob_of_appearing=0.1)
        ],
        allowed_obstacle_appearance_columns=[{2}, {1}],
        allow_crashing=True)


def safety_info(root_probs,
                transitions,
                sa_safety_info,
                policy,
                discount=0.99,
                normalize=True):
    '''Assumes the first dimension is a batch dimension.'''
    state_safety_info = dual_state_value_policy_evaluation_op(
        transitions, policy, sa_safety_info, gamma=discount)

    if len(state_safety_info.shape) < 2:
        state_safety_info = tf.expand_dims(state_safety_info, 0)

    root_probs = tf.convert_to_tensor(root_probs)
    if len(root_probs.shape) < 2:
        root_probs = tf.expand_dims(root_probs, 0)

    discount = tf.convert_to_tensor(discount)
    discount = (tf.expand_dims(discount, 0)
                if len(discount.shape) == 1 else tf.transpose(discount))

    if normalize:
        state_safety_info = (1.0 - discount) * state_safety_info
    return tf.reduce_sum(root_probs * state_safety_info, axis=-1)


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


def tabular_road(headlight_range=2,
                 num_samples_per_cfr_iter=10,
                 n=100,
                 loc=0,
                 precisions=[None],
                 discount=0.99,
                 progress_bonus=1.0,
                 ditch_bonus_multiplier=3,
                 print_every=100):
    speed_limit = new_road(headlight_range=headlight_range).speed_limit()
    game = DrivingGridworld(lambda: new_road(headlight_range=headlight_range))
    num_reward_functions = n * num_samples_per_cfr_iter
    wc_ncer = fixed_ditch_bonus(
        progress_bonus, multiplier=ditch_bonus_multiplier)
    cer = critical_reward_for_fixed_ditch_bonus(progress_bonus, speed_limit,
                                                discount)

    tf.logging.info('progress_bonus: {}, wc_ncer: {}, cer: {}'.format(
        progress_bonus, wc_ncer, cer))

    reward_datasets = []
    for precision in precisions:
        random_reward_function = DebrisPerceptionReward(
            stopping_reward=tf.zeros([num_reward_functions]),
            wc_non_critical_error_reward=tf.fill([num_reward_functions],
                                                 wc_ncer),
            bc_unobstructed_progress_reward=tf.fill([num_reward_functions],
                                                    progress_bonus),
            num_samples=num_reward_functions,
            critical_error_reward=tf.fill([num_reward_functions], cer),
            use_slow_collision_as_offroad_base=False,
            loc=loc,
            precision=precision)

        transitions, rfd_list, state_indices = game.road.tabulate(
            random_reward_function, print_every=print_every)

        reward_datasets.append(tf.transpose(tf.stack(rfd_list), [2, 0, 1]))

    transitions = tf.stack(transitions)
    root_probs = tf.one_hot(
        state_indices[game.road.copy().to_key()], depth=len(state_indices))
    return root_probs, transitions, reward_datasets, state_indices
