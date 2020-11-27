import numpy as np
from offpolicy_eval import offpolicy_eval_wis


def compute_transitions_and_rewards(data, n_states, n_actions):

    transition = np.zeros((n_states, n_states, n_actions))
    sums = np.zeros((n_states, n_actions))
    R = np.zeros((n_states, n_actions))

    for index, row in data.iterrows():
        if index < len(data) - 1 and data.at[index + 1, 't'] != 0:
            S0, S1, action, Y = int(data.at[index, 'St']), int(data.at[index + 1, 'St']), int(
                data.at[index, 'Xt']), int(data.at[index, 'Yt'])
            transition[S0][S1][action] += 1
            R[S0][action] += Y
            sums[S0][action] += 1

    for i in range(n_states):
        for j in range(n_actions):
            if sums[i][j] != 0:
                R[i][j] = R[i][j] / sums[i][j]
                for k in range(n_states):
                    transition[i][k][j] = transition[i][k][j] / sums[i][j]

    return transition, R, sums


def compute_state_action_visits(data, n_states, n_actions):
    sums = np.zeros((n_states, n_actions))
    for index, row in data.iterrows():
        S0, action = int(data.at[index, 'St']), int(data.at[index, 'Xt'])
        sums[S0][action] += 1
    return sums


def evaluate_policy(data, policy, n_states, n_actions, gamma, n_iters):
    data_copy = data.copy()
    sums = compute_state_action_visits(data, n_states, n_actions)
    physpol = (sums.T / ((sums.sum(axis=1) == 0) + (sums.sum(axis=1)))).T
    # behaviour policy is the physician's policy
    p = 0.01
    softpi = physpol.copy()
    for i in range(n_states):
        zeros = softpi[i] == 0
        if zeros.sum() > 0 and (~zeros).sum() > 0:
            z = p / zeros.sum()
            softpi[i][zeros] = z
            nz = p / (~zeros).sum()
            softpi[i][~zeros] = softpi[i][~zeros] - nz
    # evaluation policy is epsilon greedy version of OptimalAction
    softb = np.zeros((n_states, n_actions)) + p / (n_actions - 1)
    for i in range(n_states):
        softb[i][int(policy[i])] = 1 - p
    for index, row in data_copy.iterrows():
        data_copy.at[index, 'softpi(s,a)'] = softpi[int(row['St'])][int(row['Xt'])]
        data_copy.at[index, 'softb(s,a)'] = softb[int(row['St'])][int(row['Xt'])]
        data_copy.at[index, 'optimal_action'] = policy[int(row['St'])]

    bootwis = offpolicy_eval_wis(data_copy, gamma, n_iters)

    return data_copy, bootwis