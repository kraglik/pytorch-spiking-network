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

    def forward(self, dt=0.5):
        self.v_i = self.v_i.clamp(0, 1).ceil()

        self.trace_tc -= dt * self.trace_tc * self.trace_tc
        self.trace_tc += self.v_i
        self.trace_tc = self.trace_tc.clamp(0, 1)

        for connection in self.outputs:
            connection.forward(self.v_i, dt)


class IzhikevichGroup(nn.Module):
    def __init__(self, name, size, c=-65.0, u=-14.0, a=0.02, b=0.2, d=2.0, threshold=35.0, trace_tc=0.05):
        super(IzhikevichGroup, self).__init__()
        self.name = name

        # State of neuronal group
        self.v = torch.FloatTensor(size).zero_() + c  # Tensor of membrane potentials
        self.u = torch.FloatTensor(size).zero_() + u  # Tensor of recovery variables
        self.spike_traces = torch.FloatTensor(size).zero_()
        self.spikes = torch.FloatTensor(size).zero_()
        self.trace_tc = trace_tc
        self.t = 0.0

        # Input of neuronal group
        self.v_i = torch.FloatTensor(size).zero_()
        self.v_i_next = torch.FloatTensor(size).zero_()

        # Constant coefficients
        self.size = size            # Number of neurons in neuronal group
        self.a = a                  # Time scale of the recovery variable
        self.b = b                  # Sensitivity of the recovery variable to the subthreshold fluctuations
        self.d = d                  # After-spike reset of the recovery variable
        self.c = c                  # After-spike reset value of membrane potential
        self.threshold = threshold  # Threshold for spike generation

        # Synaptic connections
        self.inputs = []
        self.outputs = []

        self.plasticity = True

    def forward(self, dt=1.0):
        for _ in range(int(dt / 0.5)):
            self.v += dt * (0.04 * (self.v ** 2) + 5.0 * self.v + 140 - self.u + self.v_i)
            self.u += dt * self.a * (self.b * self.v - self.u)

        spikes = (self.v - self.threshold).clamp(0.0, 1.0).ceil()
        non_spikes = (1.0 - spikes)

        self.v = self.v * non_spikes + self.c * spikes
        self.u += self.d * spikes

        self.spike_traces -= dt * self.trace_tc * self.spike_traces
        self.spikes = spikes

        for connection in self.outputs:
            connection.forward(spikes, self.t)

        self.t += dt
        self.spike_traces = (self.spike_traces + spikes).clamp(0, 1.0)

        return spikes

    def swap_inputs(self):
        self.v_i, self.v_i_next = self.v_i_next, self.v_i.zero_()

    def set_plasticity(self, plasticity):
        for connection in self.inputs:
            connection.plasticity = plasticity
        for connection in self.outputs:
            connection.plasticity = plasticity
