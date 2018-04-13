import torch
import torch.nn as nn


class InputGroup(nn.Module):
    def __init__(self, size, trace_tc=0.05):
        super(InputGroup, self).__init__()
        self.size = size
        self.v_i = torch.FloatTensor(size).zero_()

        self.outputs = []
        self.trace_tc = trace_tc
        self.spike_traces = torch.FloatTensor(size).zero_()

    def forward(self):
        self.v_i = self.v_i.clamp(0, 1).ceil()

        self.trace_tc -= self.trace_tc * self.trace_tc
        self.trace_tc += self.v_i
        self.trace_tc = self.trace_tc.clamp(0, 1)

        for connection in self.outputs:
            connection.forward(self.v_i)


class IzhikevichGroup(nn.Module):
    """
    Neuronal group of Izhikevich neurons (Simple Model of Spiking Neurons, Izhikevich, 2003)

            v' = dt * (0.04v^2 + 5v + 140 - u + I)
            u' = dt * a(bv - u)

    """
    def __init__(self, name, size, c=-65.0, u=-14.0, a=0.02, b=0.2, d=8.0, threshold=35.0, trace_decay_speed=0.05):
        super(IzhikevichGroup, self).__init__()
        self.name = name

        # Internal state of neuronal group
        self.v = torch.FloatTensor(size).zero_() + c        # Tensor of membrane potentials
        self.u = torch.FloatTensor(size).zero_() + u        # Tensor of recovery variables
        self.spike_traces = torch.FloatTensor(size).zero_()
        self.spikes = torch.FloatTensor(size).zero_()
        self.trace_decay_speed = trace_decay_speed
        self.t = 0.0

        # External input
        self.v_i = torch.FloatTensor(size).zero_()
        self.v_i_next = torch.FloatTensor(size).zero_()

        # Model coefficients
        self.size = size            # Number of neurons in neuronal group
        self.a = a                  # Time scale of the recovery variable
        self.b = b                  # Sensitivity of the recovery variable to the subthreshold fluctuations
        self.d = d                  # After-spike reset of the recovery variable
        self.c = c                  # After-spike reset value of membrane potential
        self.threshold = threshold  # Threshold for spike generation

        # Synaptic connections
        self.inputs = []
        self.outputs = []

    def forward(self):
        v, u, a, b, c, d, v_max = self.v, self.u, self.a, self.b, self.c, self.d, self.threshold

        # Calculating new values with time step dt = 0.5 for numerical stability
        v += 0.5 * ((0.04 * v + 5.0) * v + 140 - u + self.v_i)
        u += 0.5 * a * (b * v - u)

        v += 0.5 * ((0.04 * v + 5.0) * v + 140 - u + self.v_i)
        u += 0.5 * a * (b * v - u)

        spikes = (v - v_max).clamp(0.0, 1.0).ceil()
        non_spikes = (1.0 - spikes)

        self.v = v * non_spikes + c * spikes    # Resetting membrane potential of fired neurons
        self.u = u + d * spikes                 # Resetting recovery variable values of fired neurons

        self.spike_traces -= self.trace_decay_speed * self.spike_traces
        self.spike_traces = (self.spike_traces + spikes).clamp(0, 1.0)
        self.spikes = spikes

        self.t += 1.0

        for connection in self.outputs:
            connection.forward(spikes)

        return spikes

    def swap_inputs(self):
        self.v_i, self.v_i_next = self.v_i_next, self.v_i.zero_()

    def set_plasticity(self, plasticity):
        for connection in self.inputs:
            connection.plasticity = plasticity
        for connection in self.outputs:
            connection.plasticity = plasticity
