import tensorflow as tf
import numpy as np
from decimal import Decimal
from driving_gridworld.human_ui import observation_to_img
from driving_gridworld.matplotlib import plot_frame_with_text
from driving_gridworld.rewards import DeterministicReward
from driving_gridworld.actions import ACTIONS


def state_idx_map(transitions_dict):
    state_indices = {}
    idx = 0
    for state in transitions_dict.keys():
        state_indices[state] = idx
        idx += 1
    assert len(state_indices) == len(transitions_dict)
    assert idx == len(transitions_dict)
    return state_indices


def test_state_idx_map(transitions):
    _patient = state_idx_map(transitions)
    assert len(_patient) == len(transitions)
    assert len(_patient) == len(transitions)

    max_idx = len(_patient)
    _patient = {
        k: tf.one_hot(idx, depth=max_idx)
        for k, idx in _patient.items()
    }
    assert len(_patient) == len(transitions)


def sample_D(root, actions):
    D = {}
    NV = [root]
    while NV:
        s = NV.pop()
        s_key = s.to_key()
        if s_key not in D:
            D[s_key] = (s, [[] for i in range(len(ACTIONS))])
            for a in actions:
                sum_probs = 0.0
                for i, (s_prime, p) in enumerate(s.successors(a)):
                    s_prime_key = s_prime.to_key()
                    D[s_key][1][a].append((s_prime, p))
                    sum_probs += p
                    if s_prime_key not in D:
                        NV.append(s_prime)
                assert 1.01 >= sum_probs >= 0.99
    return D


def sample_list_of_worlds(headlight_range, num_samples=100):
    list_rewards = []
    for i in range(num_samples):
        reward_function = DeterministicReward.sample_unshifted(
            headlight_range + 1, reward_for_critical_error=-1000.0)
        list_rewards.append(reward_function)
    return list_rewards


def sample_rewards_tensor(D_values, list_rewards):
    R = []
    for reward_function in list_rewards:
        R.append([])

    for s, actions_list in D_values:
        R[-1].append([])

        for action_idx in range(len(actions_list)):
            expected_reward = 0
            for s_p, p in actions_list[action_idx]:
                expected_reward += p * reward_function(s, action_idx, s_p)
            R[-1][-1].append(expected_reward)

    return tf.transpose(tf.stack(R), [1, 2, 0])


def generate_transitions_and_rewards(root,
                                     actions,
                                     reward_function,
                                     transitions=None,
                                     rewards=None,
                                     prevent_bad_acceleration=False):
    if transitions is None:
        transitions = {}
    if rewards is None:
        rewards = {}

    NV = [root]
    while NV:
        s = NV.pop()
        s_key = s.to_key()
        if s_key not in transitions:
            transitions[s_key] = {}
            rewards[s_key] = {}

            for a in actions:
                transitions[s_key][a] = {}
                rewards[s_key][a] = {}

                sum_probs = 0.0
                for i, (s_prime, p) in enumerate(s.successors(a)):
                    s_prime_key = s_prime.to_key()
                    transitions[s_key][a][s_prime_key] = p
                    sum_probs += p
                    rewards[s_key][a][s_prime_key] = reward_function(
                        s, a, s_prime)
                    if s_prime_key not in transitions:
                        NV.append(s_prime)

                assert i + 1 == len(rewards[s_key][a])
                assert 1.01 >= sum_probs >= 0.99
    return transitions, rewards


def action_value_iteration_step(transitions,
                                rewards,
                                gamma,
                                q_next=None,
                                q_prev=None):
    if q_next is None:
        q_next = {}
    if q_prev is None:
        q_prev = {}

    for state_tuple in transitions.keys():
        actions = sorted(transitions[state_tuple].keys())
        if state_tuple not in q_next:
            q_next[state_tuple] = np.zeros([len(actions)])

        for action_id in range(len(actions)):
            action = actions[action_id]
            sum_succ = 0.0

            for next_state in transitions[state_tuple][action].keys():
                p = transitions[state_tuple][action][next_state]
                r = rewards[state_tuple][action][next_state]
                f = q_prev[next_state].max() if next_state in q_prev else 0.0
                sum_succ += p * (r + gamma * f)

            q_next[state_tuple][action_id] = sum_succ

    return q_next


def compute_q_value(transitions,
                    rewards,
                    gamma,
                    road,
                    num_iterations=350,
                    epsilon=1e-20):
    q = {}
    q_initial = 0.0
    for t in range(num_iterations):
        q = action_value_iteration_step(
            transitions, rewards, gamma, q_prev=q, q_next=q)
        if abs(max(q[road.to_key()]) - q_initial) < epsilon:
            print(
                "q value at initial state has converged to a precision of: %.1E"
                % Decimal(epsilon))
            print("The number of iterations is = {}".format(t))
            return q
        q_initial = max(q[road.to_key()])
    print(
        "Number of iterations for the q_value is = {}, and q_value did not converge to a precision of: {:.1g} ".
        format(t, Decimal(epsilon)))
    return q


def rollout(policy, game, num_steps=100):
    observation, _, d = game.its_showtime()
    img = observation_to_img(observation)
    r = 0
    a = 4
    discounted_return = 0

    frame, ax_texts, fig, ax = plot_frame_with_text(img, r, discounted_return,
                                                    a)
    frames = [[frame] + ax_texts]
    actions = []
    rewards = []
    discounts = []

    for t in range(num_steps):
        discounts.append(d)
        a = policy(game.road)
        observation, r, d = game.play(a)
        actions.append(a)
        rewards.append(r)

        if d == 0: break

        discounted_return += (d**t) * r

        frame, ax_texts, fig, ax = plot_frame_with_text(
            observation_to_img(observation),
            r,
            discounted_return,
            a,
            fig=fig,
            ax=ax)
        frames.append([frame] + ax_texts)
        return frames, fig, ax, actions, np.array(rewards), np.array(discounts)


if __name__ == '__main__':
    raise NotImplementedError()
    # test_state_idx_map(transitions)
