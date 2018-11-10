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

from robust_offline_contextual_bandits import cache


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
    def train_env(cls, root_probs, transitions, reward_dataset, discount,
                  kofn_opponent):
        num_samples = reward_dataset.shape[0].value
        n = reward_dataset.shape[1].value
        transitions = tf.convert_to_tensor(transitions)
        num_states = transitions.shape[0].value
        num_actions = transitions.shape[1].value
        r = tf.resahpe(reward_dataset,
                       [num_samples * n, num_states, num_actions])

        def env(policy):
            '''sample * world X state'''
            v = dual_state_value_policy_evaluation_op(
                transitions, policy, r, gamma=discount)
            q = tf.reshape(
                r + discount * tf.tensordot(v, transitions, axes=[-1, -1]),
                [num_samples, n, num_states, num_actions])
            '''sample X state X action X world'''
            q = tf.transpose(q, [0, 2, 3, 1])
            v = tf.transpose(
                tf.reshape(v, [num_samples, n, num_states]), [0, 2, 1])

            kofn_q = [
                kofn_action_values(
                    q[sample_idx],
                    KofnEvsAndWeights(
                        v[sample_idx],
                        kofn_opponent,
                        context_weights=root_probs).world_weights)
                for sample_idx in range(num_samples)
            ]
            return tf.reduce_mean(tf.stack(kofn_q, -1), axis=-1)

        return env

    @classmethod
    def test_env(cls, root_probs, transitions, reward_dataset, discount,
                 kofn_opponent):
        num_samples = reward_dataset.shape[0].value
        n = reward_dataset.shape[1].value
        transitions = tf.convert_to_tensor(transitions)
        num_states = transitions.shape[0].value
        num_actions = transitions.shape[1].value
        r = tf.resahpe(reward_dataset,
                       [num_samples * n, num_states, num_actions])

        def env(policy):
            '''sample * world X state'''
            v = dual_state_value_policy_evaluation_op(
                transitions, policy, r, gamma=discount)
            v = tf.transpose(
                tf.reshape(v, [num_samples, n, num_states]), [0, 2, 1])
            kofn_ev = [
                KofnEvsAndWeights(
                    v[sample_idx], kofn_opponent,
                    context_weights=root_probs).ev
                for sample_idx in range(num_samples)
            ]
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


class TabularRoad(object):
    @classmethod
    def tabulate(cls,
                 headlight_range=2,
                 num_samples_per_cfr_iter=10,
                 n=100,
                 loc=0,
                 precisions=[None],
                 discount=0.99,
                 progress_bonus=1.0,
                 ditch_bonus_multiplier=3,
                 print_every=100):
        speed_limit = new_road(headlight_range=headlight_range).speed_limit()
        game = DrivingGridworld(
            lambda: new_road(headlight_range=headlight_range))
        num_reward_functions = n * num_samples_per_cfr_iter
        wc_ncer = fixed_ditch_bonus(
            progress_bonus, multiplier=ditch_bonus_multiplier)
        cer = critical_reward_for_fixed_ditch_bonus(progress_bonus,
                                                    speed_limit, discount)

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

            reward_datasets.append(
                tf.reshape(
                    tf.transpose(tf.stack(rfd_list), [2, 0, 1]), [
                        num_samples_per_cfr_iter, n,
                        len(state_indices),
                        len(rfd_list[0])
                    ]))

        transitions = tf.stack(transitions)
        root_probs = tf.one_hot(
            state_indices[game.road.copy().to_key()], depth=len(state_indices))
        return [
            cls(game, root_probs, transitions, r, discount, state_indices)
            for r in reward_datasets
        ]

    def __init__(self, game, root_probs, transitions, reward_dataset, discount,
                 state_indices):
        self.game = game
        self.root_probs = tf.convert_to_tensor(root_probs)
        self.transitions = tf.convert_to_tensor(transitions)
        self.reward_dataset = tf.convert_to_tensor(reward_dataset)
        self.state_indices = state_indices
        self.discount = discount

    @property
    def num_reps(self):
        return self.reward_dataset.shape[0].value

    @property
    def num_worlds(self):
        return self.reward_dataset.shape[1].value

    @property
    def num_states(self):
        return len(self.state_indices)

    @property
    def num_actions(self):
        return self.transitions.shape[1].value

    @property
    def num_state_actions(self):
        return self.num_states * self.num_actions

    @cache
    def sa_safety_info(self):
        sasp_safety_info, _si = self.game.road.safety_information()
        for k, v in _si.items():
            assert self.state_indices[k] == v
        sasp_safety_info = tf.stack(sasp_safety_info)

        sa_safety_info = tf.transpose(
            tf.reduce_sum(
                sasp_safety_info * tf.expand_dims(self.transitions, axis=-1),
                axis=2), [2, 0, 1])
        return sa_safety_info

    @cache
    def discount_vector(self):
        terminal_states = tf.greater(self.sa_safety_info[0, :, -1], 0)
        discount = tf.expand_dims(
            tf.where(terminal_states, tf.zeros([self.num_states]),
                     tf.fill([self.num_states], self.discount)), -1)

        num_terminal_states = tf.reduce_sum(
            tf.cast(tf.greater(self.sa_safety_info[0, :, -1], 0),
                    tf.float32)).eval()
        assert num_terminal_states == 1
        return discount
