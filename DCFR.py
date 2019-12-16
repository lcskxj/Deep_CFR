from Graph import Graph
import numpy as np
import datetime
import torch
import Sample_CFR as cfr
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import logging

Training_iteration = 1000  # number of iteration when training
LearningRate_adv = 0.01  # learning rate of advantage network
CFR_Iteration = 100  # number of iteration in CFR
N_traversal = 10  # number of sampling the game tree
n_player = 3
Horizon = 5
n_police = 2
BATCH_SIZE = 16

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
        x = F.relu(self.hidden2(x))
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
        strategy = get_strategy(player.adv_model, info)
        for i, a in enumerate(info.action):
            next_history = history[:]
            next_history.append(a)
            action_utils[i] = cfr_traversal(next_history, player, opponent, t, info_set, graph, pr_1 * strategy[i], pr_2)
        util = sum(action_utils * strategy)
        regret = (action_utils - util) * pr_2
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
        regret[i] = model(data)
        #print("djf", regret[i])
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
        data = np.zeros((Horizon - 1) * 2 + n_police + 1)
        j = 0
        # print(info.mark)
        for i in info.mark:
            for z in range(len(i)):
                data[j] = i[z]
                j = j + 1
        for z in range(len(action)):
            data[(Horizon - 1) * 2 - 1 + z] = action[z]
        data[(Horizon - 1) * 2 + n_police] = info.length
    else:
        data = np.zeros(n_player + 2)
        data[0] = info.mark[0]
        j = 1
        for i in info.mark[1]:
            data[j] = i
            j = j + 1
        data[j] = action
        data[n_player + 1] = info.length
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
        self.adv_model = self._build_net()
        self.strategy_model = self._build_net()
        self.adv_memory = []
        self.strategy_memory = []

    # build model for player
    def _build_net(self):
        if self.player_id == 0:
            model = Net_0(n_player + 2, 50, 1)
        else:
            model = Net_1((Horizon - 1) * 2 + n_police + 1, 50, 1)
        #model.apply(self.init_weights)
        return model

    '''def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            print('************reset weights**************')
            y = m.in_features
            m.weight.data.normal_(0.0, 1 / np.sqrt(y))
            m.bias.data.fill_(0)'''

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
    def train_network(self, Flag):
        if Flag == True:
            train_data = [i[0] for i in self.adv_memory]
            label_data = [i[1:] for i in self.adv_memory]
            train_data = torch.tensor(train_data)
            label_data = torch.tensor(label_data)
            torch_dataset = Data.TensorDataset(train_data, label_data)
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            for t in range(Training_iteration):
                for step, (batch_x, batch_y) in enumerate(loader):
                    out = self.adv_model(batch_x)
                    criterion = My_loss()
                    loss = criterion(out, batch_y)
                    # loss = my_loss(out, batch_y)
                    optimizer = torch.optim.SGD(self.adv_model.parameters(), lr=LearningRate_adv)
                    optimizer.zero_grad()
                    loss.backward()
                    #print(self.adv_model.hidden1.weight.grad)
                    #print(self.adv_model.hidden1.bias.grad)
                    optimizer.step()
                    print('player id:', self.player_id, 'interation:', t, '|batch step: ', step, '|loss: ', loss.item())
            #return train_data, label_data
        else:
            train_data = [i[0] for i in self.strategy_memory]
            label_data = [i[1:] for i in self.strategy_memory]
            train_data = torch.tensor(train_data)
            label_data = torch.tensor(label_data)
            torch_dataset = Data.TensorDataset(train_data, label_data)
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            for t in range(Training_iteration):
                for step, (batch_x, batch_y) in enumerate(loader):
                    out = self.strategy_model(batch_x)
                    criterion = My_loss()
                    loss = criterion(out, batch_y)
                    # loss = my_loss(out, batch_y)
                    optimizer = torch.optim.SGD(self.adv_model.parameters(), lr=LearningRate_adv)
                    optimizer.zero_grad()
                    loss.backward()
                    # print(self.adv_model.hidden1.weight.grad)
                    # print(self.adv_model.hidden1.bias.grad)
                    optimizer.step()
                    print('player id:', self.player_id, 'interation:', t, '|batch step: ', step, '|loss: ', loss.item())
            # return train_data, label_data


    def evaluate_model(self, train_data, label_data):
        # for p in self.adv_model.parameters():
        #     print(p)
        out = self.adv_model(train_data)
        out = out.squeeze(1)
        out = out.detach().numpy()
        label = label_data[:, 1].numpy()
        x = list(range(0, len(label)))
        plt.scatter(x, out, color="blue")
        plt.scatter(x, label, color="red")
        plt.show()

def main():
    graph = Graph(length=3, width=3)
    info_set = {}
    history = [2, (1, 6)]
    player_list = [Player(0), Player(1)]
    utility = []
    start_time1 = datetime.datetime.now()
    for t in range(CFR_Iteration):
        for player in player_list:
            opponent = player_list[1 - player.player_id]
            for k in range(N_traversal):
                cfr_traversal(history, player, opponent, t, info_set, graph)
            #player.adv_model.apply(player.init_weights)
            player.train_network(True)
            # player.evaluate_model(train_data, label)
        for player in player_list:
            #player.strategy_model.apply(player.init_weights)
            player.train_network(False)
        utility.append(real_play(player_list[0].strategy_model, player_list[1].strategy_model, history, graph, info_set))
        end_time1 = datetime.datetime.now()
        print("utility: ", utility, "time: ", end_time1 - start_time1)
        logging.info("Iteration:{}, utility:{}, time:{} ".format(t, utility, end_time1 - start_time1))
        exit(0)
        #player.evaluate_model()


# for param in player.model.parameters():
# 	print(param.data)


if __name__ == '__main__':
    main()

