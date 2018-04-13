import torch
import torch.nn as nn


class Connection(nn.Module):
    """
    STDP synaptic connection
    """

    def __init__(self, pre, post, latency: 1, p=0.5, nu_pre=1e-4, nu_post=1e-2, limit=1.0, plasticity=True):
        super(Connection, self).__init__()

        self.pre = pre                  # Presynaptic neuronal group
        self.post = post                # Postsynaptic neuronal group
        self.pre.outputs.append(self)
        self.post.inputs.append(self)
        self.nu_pre = nu_pre            # LTP Coefficient
        self.nu_post = nu_post          # LTD Coefficient
        self.limit = limit              # Maximal weight value
        self.latency = max(1, latency)  # Synaptic latency
        self.plasticity = plasticity    # Defines if synaptic connection's plasticity enabled
        self.w = (torch.rand(pre.size, post.size).float() - (1.0 - p)).clamp(0.0, 1.0)  # Synaptic weights

        self.traces = [
            torch.FloatTensor(pre.size).zero_() for _ in range(self.latency)
        ]
        self.spikes = [
            torch.FloatTensor(pre.size).zero_() for _ in range(self.latency)
        ]

    def forward(self, pre_spikes):
        spikes = self.spikes[-1]    # Spikes that arrived with latency
        traces = self.traces[-1]    # Traces of spikes that arrived with latency

        self.spikes.pop()
        self.traces.pop()

        self.spikes = [pre_spikes] + self.spikes               # Inserting spikes from presynaptic group to queue
        self.traces = [self.pre.spike_traces] + self.traces    # Inserting traces from presynaptic group to queue

        output = (self.w.t() * spikes).sum(1)   # Calculating input current for postsynaptic group at this moment
        self.post.v_i_next += output            # Adding input current from this connection to postsynaptic group

        if self.plasticity:                     # If synaptic plasticity if enabled
            self.update(traces, spikes)         # Updating weights

        return output

    def update(self, traces, spikes):
        pre, post = self.pre, self.post

        self.w += self.nu_post * (traces.view(pre.size, 1) * post.spikes.view(1, post.size))        # LTP
        self.w -= self.nu_pre * (spikes.view(pre.size, 1) * post.spike_traces.view(1, post.size))   # LTD
        self.w = torch.clamp(self.w, 0, self.limit)

