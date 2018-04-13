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
        self.latency = max(1, latency)
        self.plasticity = plasticity
        self.w = (torch.rand(pre.size, post.size).float() - (1.0 - p)).w.clamp(0.0, 1.0)  # Synaptic weights

        self.traces = [
            torch.FloatTensor(pre.size).zero_() for _ in range(self.latency)
        ]
        self.spikes = [
            torch.FloatTensor(pre.size).zero_() for _ in range(self.latency)
        ]

    def forward(self, pre_spikes, dt):
        spikes = self.spikes[-1]
        traces = self.traces[-1]
        self.spikes = [pre_spikes] + self.spikes[:-1]
        self.traces = [self.pre.spike_traces] + self.traces[:-1]

        output = (self.w.t() * spikes).sum(1)
        self.post.v_i_next += output

        if self.plasticity:
            self.update(traces, spikes)

        return output

    def update(self, traces, spikes):
        pre, post = self.pre, self.post

        self.w += self.nu_post * (traces.view(pre.size, 1) * post.spikes.view(1, post.size))
        self.w -= self.nu_pre * (spikes.view(pre.size, 1) * post.spike_traces.view(1, post.size))
        self.w = torch.clamp(self.w, 0, self.limit)

