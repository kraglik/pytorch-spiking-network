import torch.nn as nn
import pickle


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.groups = dict()
        self.activity_log = dict()
        self.t = 0.0

    def add(self, group):
        self.groups[group.name] = group

    def set_observable(self, group: str):
        if group not in self.groups.keys():
            raise KeyError('Group with that name not found.')
        self.activity_log[group] = []

    def step(self, inputs: dict):
        for group, input in inputs.items():
            if group in self.groups.keys():
                self.groups[group].v_i += input

        for name, group in self.groups.items():
            spikes = group.forward()
            if name in self.activity_log.keys():
                self.activity_log[name].append(spikes)

        for group in self.groups.values():
            group.swap_inputs()

        self.t += 1.0

    def set_plasticity(self, plasticity):
        for group in self.groups.values():
            group.set_plasticity(plasticity)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            net = pickle.load(f)
        return net

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

