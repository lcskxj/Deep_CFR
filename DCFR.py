from Graph import Graph
import numpy as np
import datetime
import torch
import Sample_CFR as cfr
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

Training_iteration = 1000  # number of iteration when training
LearningRate_adv = 0.001  # learning rate of advantage network
CFR_Iteration = 10  # number of iteration in CFR
N_traversal = 10  # number of sampling the game tree
n_player = 3
Horizon = 3
n_police = 2
BATCH_SIZE = 16


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


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x

class Net_1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x

# traverse the game tree, collect the training data
def cfr_traversal(history, player, opponent, t, info_set, graph, pr_1=1., pr_2=1.):
    if cfr.is_terminal(history):
        return cfr.terminal_util(history, player.player_id)
    info = cfr.get_information_set(info_set, history, graph)

    if info.player == player.player_id:
        action_utils = np.zeros(info.n_actions)
        strategy = get_strategy(player.model, info)
        for i, a in enumerate(info.action):
            next_history = history[:]
            next_history.append(a)
            action_utils[i] = cfr_traversal(next_history, player, opponent, t, info_set, graph, pr_1 * strategy[i],
                                            pr_2)
        util = sum(action_utils * strategy)
        regret = (action_utils - util) * pr_2
        player.adv_memory_add(info, t + 1, regret)
        return util
    else:
        strategy = get_strategy(opponent.model, info)
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
        regret[i] = model(data)

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
def connect(info, action_list):
    if info.player == 1:
        data = np.zeros(Horizon * 2 + n_police)

        j = 0
        for mark in info.mark:
            for z in range(len(mark)):
                data[j] = mark[z]
                j += 1

        # policy
        for z in range(len(action_list)):
            data[Horizon * 2 - 1 + z] = action_list[z]
    else:
        data = np.zeros(n_player + 1)
        data[0] = info.mark[0]
        j = 1
        for mark in info.mark[1]:
            data[j] = mark
            j += 1
        data[j] = action_list
    return data


# evaluation
def real_play(model_1, model_2, history, graph, info_set):
    if cfr.is_terminal(history):
        return cfr.terminal_util(history, 1)
    else:
        n = len(history)
        is_player = n % 2
        info = cfr.get_information_set(info_set, history, graph)
        if is_player == 0:
            strategy = get_strategy(model_1, info)
            v = np.zeros(info.n_actions)
            utility = 0
            for a in info.action:
                p = info.action.index(a)
                next_history = history[:]
                next_history.append(a)
                v[p] = real_play(model_1, model_2, next_history, graph, info_set)
                utility += v[p] * strategy[p]
            return utility
        else:
            strategy = get_strategy(model_2, info)
            v = np.zeros(info.n_actions)
            utility = 0
            for a in info.action:
                p = info.action.index(a)
                next_history = history[:]
                next_history.append(a)
                v[p] = real_play(model_1, model_2, next_history, graph, info_set)
                utility += v[p] * strategy[p]
            return utility


class Player(object):
    def __init__(self, player_id):
        self.player_id = player_id
        self.model = self._build_net()
        self.adv_memory = []
        self.strategy_memory = []

    # build model for player
    def _build_net(self):
        if self.player_id == 0:
            model = Net(n_player + 1, 50, 1)
        else:
            model = Net(Horizon * 2 + n_police, 50, 1)

        model.apply(self.init_weights)
        return model

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    # add the data to the memory
    def adv_memory_add(self, info, t, regret):
        for i, a in enumerate(info.action):
            data = connect(info, a)
            experience = (data, t, regret[i])
            self.adv_memory.append(experience)

    # add the data to the memory
    def strategy_memory_add(self, info, t, strategy):
        for i, a in enumerate(info.action):
            data = connect(info, a)
            experience = (data, t, strategy[i])
            self.strategy_memory.append(experience)

    # train a network here, return model
    def train_network(self):
        plt_loss = []
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LearningRate_adv)
        train_data = torch.from_numpy(self.adv_memory[0][0])
        train_data = train_data.unsqueeze(0)
        target_data = torch.tensor([self.adv_memory[0][1], self.adv_memory[0][2]])
        target_data = target_data.unsqueeze(0)
        for i in range(len(self.adv_memory)):
            if i != 0:
                b = torch.from_numpy(self.adv_memory[i][0])
                b = b.unsqueeze(0)
                c = torch.tensor([self.adv_memory[i][1], self.adv_memory[i][2]])
                c = c.unsqueeze(0)
                train_data = torch.cat((train_data, b), dim=0, out=None)
                target_data = torch.cat((target_data, c), dim=0, out=None)
        torch_dataset = Data.TensorDataset(train_data, target_data)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        for t in range(Training_iteration):
            for step, (batch_x, batch_y) in enumerate(loader):
                out = self.model(batch_x)
                criterion = My_loss()
                loss = criterion(out, batch_y)
                # loss = my_loss(out, batch_y)
                optimizer.zero_grad()
                # print(self.model.hidden1.weight.grad)
                loss.backward()
                # print(self.model.hidden1.weight.grad)
                optimizer.step()
                #print('player id:', self.player_id, 'interation:', t, '|batch step: ', step, '|loss: ', loss.item())
            out = self.model(train_data)
            criterion = My_loss()
            loss = criterion(out, target_data)
            plt_loss.append(loss.item())
            print('interation:', t, '|loss: ', loss.item())
    def evaluate_model(self):
        train_data = torch.from_numpy(self.adv_memory[0][0])
        train_data = train_data.unsqueeze(0)
        target_data = torch.tensor([self.adv_memory[0][1], self.adv_memory[0][2]])
        target_data = target_data.unsqueeze(0)
        for i in range(1, len(self.adv_memory)):
            b = torch.from_numpy(self.adv_memory[i][0])
            b = b.unsqueeze(0)
            c = torch.tensor([self.adv_memory[i][1], self.adv_memory[i][2]])
            c = c.unsqueeze(0)
            train_data = torch.cat((train_data, b), dim=0, out=None)
            target_data = torch.cat((target_data, c), dim=0, out=None)
        out = self.model(train_data)
        x = [i for i in range(len(out))]
        target = []
        for i in range(len(self.adv_memory)):
            target.append(self.adv_memory[i][2])
        out = out.detach().numpy()


# plt.scatter(x, out, color="blue")
# plt.scatter(x, target, color="red")
# plt.show()


def main():
    graph = Graph(length=3, width=3)
    info_set = {}
    history = [2, (1, 6)]
    player_list = [Player(0), Player(1)]

    for t in range(CFR_Iteration):
        for player in player_list:
            opponent = player_list[1 - player.player_id]
            for k in range(N_traversal):
                cfr_traversal(history, player, opponent, t, info_set, graph)
            player.model.apply(player.init_weights)
            player.train_network()
            exit(0)
        # player.evaluate_model()


# for param in player.model.parameters():
# 	print(param.data)


if __name__ == '__main__':
    main()

