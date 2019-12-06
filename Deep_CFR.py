from Graph import Graph
import numpy as np
import datetime
import torch
import Sample_CFR as cfr
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

N_iteration = 10 #number of iteration when training
LearningRate_adv = 0.001#learning rate of advantage network
Iteration = 10 #number of iteration in CFR
N_traversal = 10 #number of sampling the game tree
n_player = 3
Horizon = 3
n_police = 2
BATCH_SIZE = 50


#define loss function
def my_loss(x, y):
    loss = 0
    for i in range(len(x)):
        loss += (y[i][0] + 1) * torch.pow((x[i] - y[i][1]), 2)
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


#train a network here, return model
def train_network(memory, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate_adv)
    plt_loss = []
    train_data = torch.from_numpy(memory[0][0])
    train_data = train_data.unsqueeze(0)
    target_data = torch.tensor([memory[0][1], memory[0][2]])
    target_data = target_data.unsqueeze(0)
    for i in range(len(memory)):
        if i != 0:
            b = torch.from_numpy(memory[i][0])
            b = b.unsqueeze(0)
            c = torch.tensor([memory[i][1], memory[i][2]])
            c = c.unsqueeze(0)
            train_data = torch.cat((train_data, b), dim=0, out=None)
            target_data = torch.cat((target_data, c), dim=0, out=None)
    torch_dataset = Data.TensorDataset(train_data, target_data)
    loader = Data.DataLoader(dataset = torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    for t in range(N_iteration):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = model(batch_x)
            loss = my_loss(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            plt_loss.append(loss.item())
            print('interation:', t, '|step: ', step, '|loss: ', loss.item())
    plt.plot(plt_loss)
    plt.show()
    return model


#traverse the game tree, collect the training data
def cfr_traversal(history, player, model_1, model_2, memory_1, memory_2, t, info_set, graph, pr_1=1., pr_2=1.):
    if cfr.is_terminal(history):
        return cfr.terminal_util(history, player)
    n = len(history)
    info = cfr.get_information_set(info_set, history, graph)
    action_utils = np.zeros(info.n_actions)
    if info.player == player:
        strategy = get_strategy(model_1, info)
        utility = 0
        for p, a in enumerate(info.action):
            next_history = history[:]
            next_history.append(a)
            action_utils[p] = cfr_traversal(next_history, player, model_1, model_2, memory_1, memory_2, t, info_set, graph, pr_1 * strategy[p], pr_2)
        util = sum(action_utils * strategy)
        regret = (action_utils - util) * pr_2
        memory_1 = memory_add(memory_1, info, t, regret)
        return util
    else:
        strategy = get_strategy(model_2, info)
        memory_2 = memory_add(memory_2, info, t, strategy)
        a, action_p = cfr.sample_action(info, strategy)
        next_history = history[:]
        next_history.append(a)
        return cfr_traversal(next_history, player, model_1, model_2, memory_1, memory_2, t, info_set, graph, pr_1, pr_2 * action_p)


#compute strategy according to model and info
def get_strategy(model, info):
    regret = np.zeros(len(info.action))
    i = 0
    for a in info.action:
        data = connect(info, a)
        data = torch.from_numpy(data)
        regret[i] = model(data)
        i = i + 1
    total = float(sum(regret))
    if total > 0:
        strategy = regret / total
    else:
        #strategy = np.zeros(info.n_actions) + 1. / float(info.n_actions)
        strategy = np.zeros(info.n_actions)
        max_regret = max(regret)
        for i, j in enumerate(regret):
            if j == max_regret:
                strategy[i] = 1
    return strategy


#connect the data info.mark and action
def connect(info, a):
    if info.player == 1:
        data = np.zeros(Horizon * 2 + n_police)
        j = 0
        for i in info.mark:
            for z in range(len(i)):
                data[j] = i[z]
                j = j + 1
        j = 0
        for z in range(len(a)):
            data[Horizon + j] = a[z]
            j = j + 1
    else:
        data = np.zeros(n_player + 1)
        data[0] = info.mark[0]
        j = 1
        for i in info.mark[1]:
            data[j] = i
            j = j + 1
        data[j] = a
    return data


#add the data to the memory
def memory_add(memory, info, t, regret):
    p = 0
    for a in info.action:
        data = connect(info, a)
        experience = (data, t, regret[p])
        p = p + 1
        memory.append(experience)
    return memory


#evaluation
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


def evaluate_model(model, memory):
    train_data = torch.from_numpy(memory[0][0])
    train_data = train_data.unsqueeze(0)
    target_data = torch.tensor([memory[0][1], memory[0][2]])
    target_data = target_data.unsqueeze(0)
    for i in range(len(memory)):
        if i != 0:
            b = torch.from_numpy(memory[i][0])
            b = b.unsqueeze(0)
            c = torch.tensor([memory[i][1], memory[i][2]])
            c = c.unsqueeze(0)
            train_data = torch.cat((train_data, b), dim=0, out=None)
            target_data = torch.cat((target_data, c), dim=0, out=None)
    out = model(train_data)
    x = [i for i in range(len(out))]
    target = []
    for i in range(len(memory)):
        target.append(memory[i][2])
    out = out.detach().numpy()
    plt.scatter(x, out, color="blue")
    plt.scatter(x, target, color="red")
    plt.show()


def main():
    length = 3
    width = 3
    graph = Graph(length, width)
    info_set = {}
    history = [2, (1, 6)]
    adv_model_1 = Net(n_player + 1, 50, 1)
    adv_model_2 = Net((Horizon) * 2 + n_police, 50, 1)
    adv_memory_1 = []
    adv_memory_2 = []
    strategy_memory_1 = []
    strategy_memory_2 =[]
    utility = []
    for t in range(Iteration):
       for k in range(N_traversal):
            cfr_traversal(history, 0, adv_model_1, adv_model_2, adv_memory_1, strategy_memory_2, t, info_set, graph)
       for k in range(N_traversal):
            cfr_traversal(history, 1, adv_model_2, adv_model_1, adv_memory_2, strategy_memory_1, t, info_set, graph)
       start_time = datetime.datetime.now()
       adv_model_1= train_network(adv_memory_1, adv_model_1)
       end_time = datetime.datetime.now()
       print("interation:{}, player 1 length of memory:{}, time:{}".format(t, len(adv_memory_1), end_time-start_time))
       start_time = datetime.datetime.now()
       adv_model_2= train_network(adv_memory_2, adv_model_2)
       end_time = datetime.datetime.now()
       print("interation:{}, player 2 length of memory:{}, time:{}".format(t, len(adv_memory_2), end_time-start_time))
       evaluate_model(adv_model_1, adv_memory_1)
       evaluate_model(adv_model_2, adv_memory_2)
       adv_model_1 = Net(n_player + 1, 50, 1)
       adv_model_2 = Net((Horizon) * 2 + n_police, 50, 1)
       #str_model_1 = Net(n_player + 1, 50, 1)
       #str_model_2 = Net((Horizon) * 2 + n_police, 50, 1)
       #str_model_1 = train_network(strategy_memory_1, str_model_1)
       #str_model_2 = train_network(strategy_memory_2, str_model_2)
       #utility.append(real_play(str_model_1, str_model_2, history, graph, info_set))
       #print("utility: ", utility)
    #print(utility)
    #plt.plot(utility)
    #plt.show()

if __name__ == '__main__':
    main()

