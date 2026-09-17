from __future__ import annotations
import torch
from torch import nn

class FlyWordPolicy(nn.Module):
    """Trainable sparse recurrent policy constrained by a fixed fly graph."""
    def __init__(self, topology, state_features=96, candidate_features=48, hidden=128, steps=4):
        super().__init__(); self.steps = steps
        self.register_buffer("sensory", torch.as_tensor(topology["sensory"], dtype=torch.long)); self.register_buffer("motor", torch.as_tensor(topology["motor"], dtype=torch.long))
        self.node_count = int(topology["source_nodes"].shape[0])
        # Dense matmul is deliberate: sparse index_add has repeated target indices
        # that cannot be exported faithfully to ONNX. The fixed mask preserves the
        # selected 110k FlyWire connections and leaves every other synapse at zero.
        adjacency = torch.zeros(self.node_count, self.node_count); mask = torch.zeros_like(adjacency)
        src, dst = torch.as_tensor(topology["src"], dtype=torch.long), torch.as_tensor(topology["dst"], dtype=torch.long)
        adjacency[src, dst] = torch.as_tensor(topology["initial_weight"], dtype=torch.float32).clamp(-3, 3) * .05; mask[src, dst] = 1
        self.edge_weight = nn.Parameter(adjacency); self.register_buffer("edge_mask", mask)
        self.edge_weight.register_hook(lambda gradient: gradient * self.edge_mask)
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_features), nn.Linear(state_features, hidden), nn.GELU(), nn.Linear(hidden, len(self.sensory)))
        self.node_bias = nn.Parameter(torch.zeros(self.node_count)); self.candidate_encoder = nn.Sequential(nn.LayerNorm(candidate_features), nn.Linear(candidate_features, hidden), nn.GELU())
        self.policy = nn.Sequential(nn.Linear(hidden + len(self.motor), hidden), nn.GELU(), nn.Linear(hidden, 1)); self.value = nn.Sequential(nn.Linear(len(self.motor), hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, state, candidates, valid, recurrent=None):
        batch = state.shape[0]; stimulus = self.state_encoder(state).float()
        h = stimulus.new_zeros(batch, self.node_count) if recurrent is None else recurrent.float()
        drive = stimulus.new_zeros(batch, self.node_count); drive[:, self.sensory] = stimulus
        for _ in range(self.steps):
            incoming = h @ self.edge_weight
            h = torch.tanh(.62 * h + incoming + drive + self.node_bias)
        motor = h[:, self.motor]; encoded = self.candidate_encoder(candidates); context = motor[:, None, :].expand(-1, candidates.shape[1], -1)
        logits = self.policy(torch.cat((encoded, context), dim=-1)).squeeze(-1).masked_fill(~valid, float("-inf"))
        return logits, self.value(motor).squeeze(-1), h
