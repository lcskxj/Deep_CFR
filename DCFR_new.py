from Graph import Graph
import numpy as np
import datetime
import torch
import Sample_CFR as cfr
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import logging
import math

plt.ion()
Training_epoch = 500  # number of epoch when training
LearningRate_adv = 0.01  # learning rate of advantage network
CFR_Iteration = 100  # number of iteration in CFR
N_traversal = 10  # number of sampling the game tree
n_player = 3
Horizon = cfr.Horizon
n_police = 2

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.INFO, format=LOG_FORMAT)


class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        t = y[:, 0].unsqueeze(0)
        y_test = y[:, 1].unsqueeze(1)
        return torch.div(torch.mm(t, torch.pow((x - y_test), 2)), len(x))


def my_loss(x, y):
    loss = 0
    for i in range(len(x)):
        loss += (y[i][0]) * torch.pow((x[i] - y[i][1]), 2)
    loss = loss / len(x)
    return loss


class Net_0(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_0, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
#       self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.hidden1(x))
        x = self.hidden2(x)
#       x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x


class Net_1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_1, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.hidden3(x)
        x = self.out(x)
        return x


# traverse the game tree, collect the training data
def cfr_traversal(history, player, opponent, t, info_set, graph, pr_1=1., pr_2=1.):
    if cfr.is_terminal(history):
        return cfr.terminal_util(history, player.player_id)
    info = cfr.get_information_set(info_set, history, graph)

    if info.player == player.player_id:
        action_utils = np.zeros(info.n_actions)
        strategy = get_strategy(player.adv_model, info)
        for i, a in enumerate(info.action):
            next_history = history[:]
            next_history.append(a)
            action_utils[i] = cfr_traversal(next_history, player, opponent, t, info_set, graph, pr_1 * strategy[i], pr_2)
        util = sum(action_utils * strategy)
        regret = action_utils - util
        player.adv_memory_add(info, t + 1, regret)
        return util
    else:
        strategy = get_strategy(opponent.adv_model, info)
        opponent.strategy_memory_add(info, t + 1, strategy)
        a, action_p = cfr.sample_action(info, strategy)
        next_history = history[:]
        next_history.append(a)
        return cfr_traversal(next_history, player, opponent, t, info_set, graph, pr_1, pr_2 * action_p)


# compute strategy according to model and info
def get_strategy(model, info):
    regret = np.zeros(len(info.action))
    for i, a in enumerate(info.action):
        data = connect(info, a)
        data = torch.from_numpy(data)
        # print(data)
        regret[i] = max(model(data), 0)
        # print("djf", regret[i])
    total = float(sum(regret))
    if total > 0.0:
        strategy = regret / total
    else:
        strategy = np.zeros(info.n_actions) + 1. / float(info.n_actions)
    # strategy = np.zeros(info.n_actions)
    # max_regret = max(regret)
    # for i, j in enumerate(regret):
    # 	if j == max_regret:
    # 		strategy[i] = 1
    return strategy


# connect the data info.mark and action
def connect(info, action):
    if info.player == 1:
        data = np.zeros((Horizon - 1) * 2 + n_police)
        j = 0
        # print(info.mark)
        for i in info.mark:
            for z in range(len(i)):
                data[j] = i[z]
                j = j + 1
        for z in range(len(action)):
            data[(Horizon - 1) * 2 + z] = action[z]
        #data[(Horizon - 1) * 2 + n_police] = info.length
    else:
        data = np.zeros((Horizon - 1) * n_player - n_police + 1)
        j = 0
        for i in info.mark:
            if isinstance(i, int):
                data[j] = i
                j = j + 1
            else:
                for z in range(len(i)):
                    data[j] = i[z]
                    j = j + 1
            data[(Horizon - 1) * n_player - n_police] = action
        #data[n_player + 1] = info.length
    return data


# evaluation
# def real_play(model_1, model_2, history, graph, info_set):
#     if cfr.is_terminal(history):
#         return cfr.terminal_util(history, 1)
#     else:
#         n = len(history)
#         is_player = n % 2
#         info = cfr.get_information_set(info_set, history, graph)
#         if is_player == 0:
#             strategy = get_strategy(model_1, info)
#             #print(strategy)
#             v = np.zeros(info.n_actions)
#             utility = 0
#             for a in info.action:
#                 p = info.action.index(a)
#                 next_history = history[:]
#                 next_history.append(a)
#                 v[p] = real_play(model_1, model_2, next_history, graph, info_set)
#                 utility += v[p] * strategy[p]
#             return utility
#         else:
#             strategy = get_strategy(model_2, info)
#             #print(strategy)
#             v = np.zeros(info.n_actions)
#             utility = 0
#             for a in info.action:
#                 p = info.action.index(a)
#                 next_history = history[:]
#                 next_history.append(a)
#                 v[p] = real_play(model_1, model_2, next_history, graph, info_set)
#                 utility += v[p] * strategy[p]
#             return utility


def real_play(history, graph, info_set):
    if cfr.is_terminal(history):
        return cfr.terminal_util(history, 1)
    else:
        n = len(history)
        info = cfr.get_information_set(info_set, history, graph)
        strategy = info.average_strategy
        v = np.zeros(info.n_actions)
        utility = 0
        for p, a in enumerate(info.action):
            next_history = history[:]
            next_history.append(a)
            v[p] = real_play(next_history, graph, info_set)
            utility += v[p] * strategy[p]
        return utility


def data_exist(data, t, memory):
    index = -1
    for i in range(len(memory)):
        #if memory[i][1] == t:
        if (memory[i][0] == data).all():
            index = i
            break
    return index


def Process_adv_memory(memory):
    new_memory = []
    for i, temp in enumerate(memory):
        data = temp[0:1]
        regret = temp[2:]
        #print(regret)
        temp = np.mean(regret)
        experience = (data, temp)
        new_memory.append(experience)
    return new_memory


class Player(object):
    def __init__(self, player_id):
        self.player_id = player_id
        self.adv_model = self._build_net()
        self.strategy_model = self._build_net()
        self.adv_memory = []
        self.strategy_memory = []

    # build model for player
    def _build_net(self):
        if self.player_id == 0:
            model = Net_0((Horizon - 1) * n_player - n_police + 1 , 50, 1)
        else:
            model = Net_1((Horizon - 1) * 2 + n_police, 50, 1)
        model.apply(self.init_weights)
        return model

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            print('reset weights......')
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)

    # add the data to the memory
    def adv_memory_add(self, info, t, regret):
        for i, a in enumerate(info.action):
            data = connect(info, a)
            index = data_exist(data, t, self.adv_memory)
            if index != -1:
                temp = [regret[i]]
                temp = tuple(temp)
                self.adv_memory[index] = self.adv_memory[index] + temp
            else:
                experience = (data, t, regret[i])
                self.adv_memory.append(experience)

    # add the data to the memory
    def strategy_memory_add(self, info, t, strategy):
        for i, a in enumerate(info.action):
            data = connect(info, a)
            experience = (data, t, strategy[i])
            self.strategy_memory.append(experience)

    # train a network here, return model
    def train_network(self, criterion, optimizer, train_adv):
        plot_loss = []
        if train_adv:
            memory = Process_adv_memory(self.adv_memory)
            #memory = self.adv_memory
            model = self.adv_model
        else:
            memory = self.strategy_memory
            model = self.strategy_model

        train_data = [i[0] for i in memory]
        label_data = [i[1:] for i in memory]
        if len(train_data) > 10000:
            batch_size = 64
        else:
            batch_size = 16

        train_data = torch.tensor(train_data)
        if train_adv:
            train_data = train_data.squeeze(1)
        label_data = torch.tensor(label_data)
        torch_dataset = Data.TensorDataset(train_data, label_data)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        for epoch in range(Training_epoch):
            for step, (batch_x, batch_y) in enumerate(loader):
                out = model(batch_x)
                #print(batch_x, out, batch_y)
                loss = criterion(out, batch_y)
                #print(loss.item())
                # loss = my_loss(out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                #print(model.hidden2.weight.grad)
                #print(model.hidden1.bias.grad)
                optimizer.step()

            out = model(train_data)
            loss = criterion(out, label_data)
            plot_loss.append(loss)
            print('player id:', self.player_id, 'epoch:', epoch, '|loss: ', loss.item())
            #print(out, train_data, label_data)

        plt.plot(plot_loss)
        plt.show()


def main():
    graph = Graph(length=3, width=3)
    info_set = {}
    history = [2, (1, 6)]
    player_list = [Player(0), Player(1)]
    utility = []
    start_time1 = datetime.datetime.now()
    for t in range(CFR_Iteration):
        print('************************************')
        print('T:', t)
        for player in player_list:
            opponent = player_list[1 - player.player_id]
            print("CFR sampling......")
            for k in range(N_traversal):
                cfr_traversal(history, player, opponent, t, info_set, graph)
            player.adv_model.apply(player.init_weights)
            criterion = torch.nn.MSELoss(reduction='elementwise_mean')
            optimizer = torch.optim.Adam(player.adv_model.parameters(), lr=LearningRate_adv)
            player.train_network(criterion, optimizer, train_adv=True)
            # player.evaluate_model(train_data, label)
        for _, info in info_set.items():
            info.strategy_sum += get_strategy(player_list[info.player].adv_model, info)
            info.average_strategy = info.get_average_strategy()
        utility.append(real_play(history, graph, info_set))
        end_time1 = datetime.datetime.now()
        print("utility: ", utility, "time: ", end_time1 - start_time1)
    plt.plot(utility)
    plt.show()
    '''if t % 5 == 0:
            for player in player_list:
                criterion = My_loss()
                player.strategy_model.apply(player.init_weights)
                optimizer = torch.optim.Adam(player.strategy_model.parameters(), lr=LearningRate_adv)
                player.train_network(criterion, optimizer, train_adv=False)
            utility.append(real_play(player_list[0].strategy_model, player_list[1].strategy_model, history, graph, info_set))
            end_time1 = datetime.datetime.now()
            print("utility: ", utility, "time: ", end_time1 - start_time1)
            logging.info("Iteration:{}, utility:{}, time:{} ".format(t, utility, end_time1 - start_time1))
        #player.evaluate_model()'''


# for param in player.model.parameters():
# 	print(param.data)


if __name__ == '__main__':
    main()
    '''info = cfr.InformationSet('sfsdf', [2, (1, 6), 3, (4, 5)], [2, 3], 2)
    a = 4
    print(connect(info, a))'''

