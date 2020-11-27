import numpy as np
import pandas as pd


def OffpolicyQlearning(data, gamma, alpha, numtraces):
    n_states = len(data['St'].unique())
    n_actions = len(data['Xt'].unique())
    sumQ = np.zeros(numtraces)
    Q = np.zeros((n_states, n_actions))
    maxavgQ = 1
    modu = 100
    first_step = data[data['t'] == 0].index
    nrepi = len(first_step)
    j = 0
    done = False
    while j < numtraces and not done:
        i = first_step[np.random.randint(nrepi - 1)]  # pick one episode randomly (not the last one!)
        trace = pd.DataFrame()

        while data.at[i + 1, 't'] != 0:
            s_a_r = {'S1': data.at[i + 1, 'St'],
                     'a1': data.at[i + 1, 'Xt'],
                     'r1': data.at[i + 1, 'Yt']}
            trace = trace.append(s_a_r, ignore_index=True)
            i += 1
        tracelength = len(trace)
        return_t = trace.at[tracelength - 1, 'r1']  # get last reward as return for penultimate state and action
        for t in range(tracelength - 2, -1, -1):
            s, a = int(trace.at[t, 'S1']), int(trace.at[t, 'a1'])
            Q[s][a] = (1 - alpha) * Q[s][a] + alpha * return_t
            return_t = return_t * gamma + trace.at[t, 'r1']

        sumQ[j] = Q.sum()
        j = j + 1
        if j > 0 and (j % modu * 500) == 0:
            s = sumQ[j - modu * 500:].mean()
            if maxavgQ == 0:
                d = np.nan
            else:
                d = (s - maxavgQ) / maxavgQ
            if abs(d) < 0.001:
                done = True
            maxavgQ = s
    return Q, sumQ


def offpolicy_eval_tdlearning(data, physpol, gamma, num_iter):
    n_states = len(physpol)
    n_actions = len(physpol[0])
    bootql = np.zeros(num_iter)
    patients = data['pt_id'].unique()
    prop = 5000 / len(patients)  #  5000 patients of the samples are used
    prop = min(prop, 0.75)  # max possible value is 0.75 (75% of the samples are used)
    a = data.loc[data['t'] == 0, 'St']
    d = np.zeros(n_states)  # initial state distribution
    for i in range(n_states):
        d[i] = (a == i).sum()
    for i in range(num_iter):
        chosen = np.floor(np.random.rand(len(patients)) + prop)
        q = data.loc[data['pt_id'].isin(patients[chosen == 1])].reset_index()
        Qoff, _ = OffpolicyQlearning(q, gamma, 0.1, 3000)

        V = np.zeros((n_states, n_actions))
        for k in range(n_states):
            for l in range(n_actions):
                V[k][l] = physpol[k][l] * Qoff[k][l]

        Vs = V.sum(axis=1)
        bootql[i] = (Vs * d).sum() / d.sum()

    return bootql


def offpolicy_eval_wis(data,gamma ,num_iter):
    bootwis = np.zeros(num_iter)
    patients = data['pt_id'].unique()
    prop = 25000/len(patients) # 25000 patients of the samples are used
    prop = min(prop, 0.75) # max possible value is 0.75 (75% of the samples are used)
    for jj in range(num_iter):
        chosen = np.floor(np.random.rand(len(patients)) + prop)
        q = data.loc[data['pt_id'].isin(patients[chosen==1])].reset_index()
        fence_posts = q.loc[q['t']==0].index
        num_of_trials = len(fence_posts)
        individual_trial_estimators = np.empty(num_of_trials)
        individual_trial_estimators[:] = np.nan
        rho_array = np.empty(num_of_trials)
        rho_array[:] = np.nan
        c = 0
        for i in range(num_of_trials - 1):
            rho = 1
            for t in range(fence_posts[i], fence_posts[i+1]-1):
                if q.at[t,'softpi(s,a)'] == 0:
                    rho = np.nan
                else:
                    rho = rho*q.at[t,'softb(s,a)']/q.at[t,'softpi(s,a)']
            if rho > 0:
                c += 1
            rho_array[i] = rho
        normalization = np.nansum(rho_array)
        for i in range(num_of_trials - 1):
            current_trial_estimator = 0
            rho = 1
            discount = 1/gamma
            for t in range(fence_posts[i], fence_posts[i+1]-1):
                if q.at[t,'softpi(s,a)'] == 0:
                    rho = np.nan
                else:
                    rho = rho*q.at[t,'softb(s,a)']/q.at[t,'softpi(s,a)']
                discount  = discount*gamma
                current_trial_estimator = current_trial_estimator + discount * q.at[t+1, 'Yt']
            individual_trial_estimators[i] =  current_trial_estimator*rho
        bootwis[jj] = np.nansum(individual_trial_estimators)/normalization
    individual_trial_estimators = (individual_trial_estimators/rho_array)[rho_array != np.nan]
    return bootwis


def offpolicy_multiple_eval(data, physpol, gamma, iter_ql, iter_wis):
    bootql = offpolicy_eval_tdlearning(data, physpol, gamma, iter_ql)
    bootwis = offpolicy_eval_wis(data, gamma, iter_wis)
    return bootql, bootwis