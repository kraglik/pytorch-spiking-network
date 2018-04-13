import torch
import numpy as np
import matplotlib.pyplot as plt
from random import random

from connection import Connection
from group import IzhikevichGroup
from network import Network


if __name__ == '__main__':
    net = Network()

    g1 = IzhikevichGroup('group 1', 500, trace_decay_speed=0.05)
    g2 = IzhikevichGroup('group 2', 500, trace_decay_speed=0.05)

    g1_to_g1 = Connection(g1, g1, latency=1, p=0.25)
    g2_to_g2 = Connection(g2, g2, latency=1, p=0.25)

    g2_to_g1 = Connection(g2, g1, latency=2, p=0.15)
    g1_to_g2 = Connection(g1, g2, latency=2, p=0.15)

    net.add(g1)
    net.add(g2)

    net.set_observable('group 1')
    net.set_observable('group 2')

    t_max = 2000.0

    plt.matshow(g1_to_g1.w.cpu().numpy())
    plt.title('Group 1 recurrent connection initial weights matrix')
    plt.savefig('plots/g1_to_g1_w_initial.png')

    print('simulation started...')

    g1_input = torch.FloatTensor(g1.size).float().zero_()
    g2_input = torch.FloatTensor(g2.size).float().zero_()

    for i in range(20):
        g2_input[i * 16 + 18] = 25.0

    for i in range(50):
        g1_input[i * 5] = 25.0

    while net.t < t_max / 2:
        inputs = {
            'group 1': g1_input + torch.rand(g1.size).float() * (random() * 10.0),
            'group 2': g2_input + torch.rand(g2.size).float() * (random() * 10.0)
        }
        net.step(inputs)

    net.set_plasticity(False)

    while net.t < t_max:
        inputs = {
            'group 1': g1_input + torch.rand(g1.size).float() * (random() * 10.0),
            'group 2': g2_input * 0.1 + torch.rand(g2.size).float() * (random() * 2.0)
        }
        net.step(inputs)

    print('done.')

    group_1_activity = torch.stack(net.activity_log['group 1']).cpu().numpy().transpose()
    group_2_activity = torch.stack(net.activity_log['group 2']).cpu().numpy().transpose()

    plt.matshow(group_1_activity)
    plt.title('Group 1 activity')
    plt.savefig('plots/group_1_activity.png')

    plt.matshow(group_2_activity)
    plt.title('Group 2 activity')
    plt.savefig('plots/group_2_activity.png')

    plt.matshow(g1_to_g1.w.cpu().numpy())
    plt.title('Group 1 recurrent connection weights matrix')
    plt.savefig('plots/g1_to_g1_w.png')

    plt.matshow(g1_to_g2.w.cpu().numpy())
    plt.title('Group 1 to Group 2 connection weights matrix')
    plt.savefig('plots/g1_to_g2_w.png')

    plt.matshow(g2_to_g1.w.cpu().numpy())
    plt.title('Group 2 to Group 1 connection weights matrix')
    plt.savefig('plots/g2_to_g1_w.png')

    plt.matshow(g2_to_g2.w.cpu().numpy())
    plt.title('Group 2 recurrent connection weights matrix')
    plt.savefig('plots/g2_to_g2_w.png')

