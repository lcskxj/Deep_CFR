from Graph import Graph
import numpy as np
import itertools
import csv
import random

Horizon = 5
Delta = 0.8
epsilon = 1
EPS = 0.001


class InformationSet(object):
    def __init__(self, key, history, action, n_actions):
        '''parameter:
        key: index of informationset
        history: the history of informationset
        action: actions under this informationset
        n_action: the number of action
        reach_pr: the probability of reaching this information set'''
        self.player = len(history) % 2
        self.length = int(len(history) / 2)
        self.key = key
        self.history = [history]
        self.n_actions = n_actions
        self.action = action
        self.regret_sum = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions) + 1. / float(n_actions)
        self.strategy = np.zeros(n_actions) + 1. / float(n_actions)
        self.average_strategy = np.zeros(n_actions)
        self.reach_pr = 0.
        if self.player == 0:
            self.mark = history[:len(history) - 1]
        else:
            self.mark = []
            for j in range(1, len(history), 2):
                self.mark.append(history[j])

    def get_average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            strategy = self.strategy_sum / float(total)
            strategy = np.where(strategy < EPS, 0., strategy)
            total = sum(strategy)
            strategy /= float(total)
        else:
            strategy = np.zeros(self.n_actions) + 1. / float(self.n_actions)
        return strategy


#judge if the history1 is in the history_list this is only for player 1 (police)
def match(history1, history_list):
    flag = 0
    temp_history = history_list
    for j in range(1, len(history1), 2):
        if history1[j] == temp_history[j]:
            flag = 1
        else:
            flag = 0
            break
    if flag == 1:
        return True
    else:
        return False


# bulid information sets(if the info of this history exists, then add the history to this info.history, otherwise build a new info and add it to info_set)
#return the info include the history
def get_information_set(info_set, history, graph):
    temp = 0
    if len(history) % 2 == 1:
        for dix, info in info_set.items():
            history_temp = info.history[0]
            if len(history) == len(history_temp):
                if match(history, history_temp):
                    info.history.append(history)
                    info1 = info
                    temp = 1
    else:
        for dix, info in info_set.items():
            history_temp = info.history[0]
            if len(history) == len(history_temp):
                if history[:len(history) - 1] == history_temp[:len(history_temp) - 1]:
                    info.history.append(history)
                    info1 = info
                    temp = 1
    if temp == 0: #if there is no information set containing history, then build one
        key = str(history)
        info1 = None
        action = next_action(graph, history)
        n_action = len(action)
        info1 = InformationSet(key, history, action, n_action)
        info_set[key] = info1
    return info1


#Return the next_action according to the history and graph
def next_action(graph, history):
    if (len(history)) % 2 == 0:
        action = history[-2]
        action_next = graph.adj[action - 1]
    else:
        action = history[-2]
        b = [0] * len(action)
        for i in range(len(action)):
            b[i] = graph.adj[action[i] - 1]
        action_next = list(itertools.product(*b))
    return action_next


# judge if the history is terminal node
def is_terminal(history):
    if len(history) / 2 >= Horizon:
        return True
    elif len(history) < 2 or len(history) % 2 == 1:
        return False
    else:
        pursuer_action = history[-1]
        evader_action = history[-2]
        if evader_action in pursuer_action:
            return True
        else:
            return False


#return utility of terminal node
def terminal_util(history, player):
    n = len(history) / 2
    pursuer_action = history[-1]
    evader_action = history[-2]
    if n <= Horizon:
        if evader_action in pursuer_action:
            if player == 0:
                return -10 * Delta ** n
            else:
                return 10 * Delta ** n
        else:
            if player == 0:
                return 10
            else:
                return -10


#sample action according to sample_strategy, return sampled action and action_probability
def sample_action(info, sample_strategy):
    temp = random.randint(1, 100000) / 100000.
    strategy_sum = 0
    for i in range(0, info.n_actions):
        strategy_sum += sample_strategy[i]
        if temp <= strategy_sum:
            action = info.action[i]
            action_probability = sample_strategy[i]
            break
        elif i == info.n_actions - 1:
            action = info.action[i]
            action_probability = sample_strategy[i]
            break
    return action, action_probability


def cfr(info_set, graph, history, pr_1=1., pr_2=1.):
    if is_terminal(history):
        utility = terminal_util(history, 1)
        return utility
    n = len(history)
    is_player_1 = n % 2 == 0
    info = get_information_set(info_set, history, graph)
    strategy = info.strategy
    if is_player_1:
        info.reach_pr += pr_1
    else:
        info.reach_pr += pr_2
    action_utils = np.zeros(info.n_actions)
    # last_history = history
    if is_player_1:
        available_action = next_action(graph, history)
        for i, action in enumerate(available_action):
            next_history = history[:]
            next_history.append(action)
            action_utils[i] = -1 * cfr(info_set, graph, next_history, pr_1 * strategy[i], pr_2)
            #print('player1', next_history, pr_1 * strategy[i], action_utils[i])
    else:
        available_action = next_action(graph, history)
        for i, action in enumerate(available_action):
            next_history = history[:]
            next_history.append(action)
            action_utils[i] = -1 * cfr(info_set, graph, next_history, pr_1, pr_2 * strategy[i])
            #print('player2', next_history, pr_2 * strategy[i], action_utils[i])
    util = sum(action_utils * strategy)
    regrets = action_utils - util
    if is_player_1:
        info.regret_sum += pr_2 * regrets
    else:
        info.regret_sum += pr_1 * regrets
    return util


def cfr_out_sample(info_set, graph, history, player, opponent_p=1., sample_p=1.):
    if is_terminal(history):
        utility = terminal_util(history, player) #/ sample_p
        #print('utility', utility, sample_p)
        return utility
    # if chance node
    n = len(history)
    is_player = n % 2 + 1
    if is_player != player:
        info = get_information_set(info_set, history, graph)
        strategy = update_strategy(info)
        for a in info.action:
            p = info.action.index(a)
            info.strategy_sum[p] += strategy[p] / sample_p
        action, action_probability = sample_action(info, strategy)
        #print('  ', action)
        next_history = history[:]
        next_history.append(action)
        return cfr_out_sample(info_set, graph, next_history, player, opponent_p * action_probability, sample_p)
    info = get_information_set(info_set, history, graph)
    strategy = update_strategy(info)
    sample_strategy = epsilon * (np.zeros(info.n_actions) + 1. / float(info.n_actions)) + (1 - epsilon) * strategy
    action, action_probability = sample_action(info, sample_strategy)
    #print(action)
    next_history = history[:]
    next_history.append(action)
    v = np.zeros(info.n_actions)
    utility = 0
    for a in info.action:
        p = info.action.index(a)
        if a == action:
            v[p] = cfr_out_sample(info_set, graph, next_history, player, opponent_p, sample_p * action_probability)
        else:
            v[p] = 0
        utility += v[p] * strategy[p]
        #print(' ', utility)
    for a in info.action:
        p = info.action.index(a)
        regret = v[p] - utility
        info.regret_sum[p] += opponent_p * regret
    #print(history, v, info.action, strategy, utility)
    return utility


#regret matching
def update_strategy(info):
    regret = np.where(info.regret_sum > 0, info.regret_sum, 0)
    total = float(sum(regret))
    if total > 0:
        strategy = regret / total
    else:
        strategy = np.zeros(info.n_actions) + 1. / float(info.n_actions)
    return strategy


def test_cfr_out_sample(info_set, graph, history):
    iteration = 10
    result_value1 = []
    result_value2 = []
    for i in range(1, iteration+1):
        result_value1 = cfr_out_sample(info_set, graph, history, 1)
        #print(result_value1)
        result_value2 = cfr_out_sample(info_set, graph, history, 2)
        #print(result_value2)
        for _, info in info_set.items():
            info.strategy_sum += info.reach_pr * info.strategy
            info.strategy = update_strategy(info)
            info.reach_pr = 0.
    count_history = 0
    cfr(info_set, graph, history)
    # show the results
    result_list = []
    #sorted(info_set.keys())
    for dix, info in info_set.items():
        total = sum(info.strategy_sum)
        info.strategy = info.strategy_sum / float(total)
        count_history += len(info.history)
        result_list.append([info.player, info.history, info.action, info.strategy])
    print(len(info_set))
    print(count_history)
    with open("result6.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)
    print('ok')


def main():
    info_set = {}
    history = [2,(3,4),5,(4,5)]
    graph = Graph(3,3)
    info = get_information_set(info_set, history, graph)
    print(info.mark)
    #  结果可视化


if __name__ == '__main__':
    main()