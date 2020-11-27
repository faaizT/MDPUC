import numpy as np


def mdp_computePpolicyPRpolicy(P, R, policy):
    n_states = len(P)
    n_actions = len(P[0][0])
    Ppolicy = np.zeros((n_states,n_states))
    PRpolicy = np.zeros(n_states)
    for a in range(n_actions):
        ind = policy==a
        if ind.sum()>0:
            for j in range(n_states):
                if ind[j]:
                    for i in range(n_states):
                        Ppolicy[j][i] = P[j][i][a]
                    PRpolicy[j] = R[j][a]
    return Ppolicy, PRpolicy


def mdp_eval_Q_matrix(P, R, discount, policy):
    n_states = len(P)
    n_actions = len(P[0][0])
    Q = np.zeros((n_states, n_actions))
    Vpolicy = mdp_eval_policy_matrix(P,R,discount,policy)
    for s0 in range(n_states):
        for a0 in range(n_actions):
            next_state_reward = 0
            for s1 in range(n_states):
                next_state_reward += P[s0][s1][a0]*Vpolicy[s1]
            Q[s0][a0] = R[s0][a0] + discount*next_state_reward
    return Q


def mdp_eval_policy_matrix(P,R,discount,policy):
    n_states = len(P)
    Ppolicy, PRpolicy = mdp_computePpolicyPRpolicy(P, R, policy)
    Vpolicy = np.matmul(np.linalg.inv(np.identity(n_states) - discount*Ppolicy), PRpolicy)
    return Vpolicy


def mdp_bellman_operator_with_Q(P, PR, discount, Vprev):
    n_states = len(P)
    n_actions = len(P[0][0])
    Q = np.zeros((n_states, n_actions))
    for a in range(n_actions):
        for i in range(n_states):
            next_step_reward = 0
            for j in range(n_states):
                next_step_reward += P[i][j][a]*Vprev[j]
            Q[i][a] = PR[i][a] + discount*next_step_reward
    V = Q.max(axis=1)
    policy = Q.argmax(axis=1)
    return V, Q, policy


def mdp_policy_iteration_with_Q(P, R, discount, policy0, max_iter=1000):
    i = 0
    policy = policy0
    is_done = False
    while not is_done:
        i += 1
        V = mdp_eval_policy_matrix(P,R,discount,policy)
        nil, Q, policy_next = mdp_bellman_operator_with_Q(P,R,discount,V)
        n_different = (policy_next != policy).sum()
        if n_different == 0 or i == max_iter or (i > 20 and n_different <= 5):
            is_done = True
        else:
            policy = policy_next
    return V, policy, i, Q