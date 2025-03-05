import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import is_oom_exception, init_rsqrt_uniform_


class GatedExperts(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models with parallel expert computation.

    Attributes:
        w1 (torch.Tensor): 3D weight tensor for input-to-hidden transformation (num_experts, dim, inter_dim).
        w2 (torch.Tensor): 3D weight tensor for hidden-to-output transformation (num_experts, inter_dim, dim).
        w3 (torch.Tensor): 3D weight tensor for feature transformation (num_experts, dim, inter_dim).
    """

    def __init__(self, n_blocks: int, num_experts: int, d_first: int, d_out: int, d_block: int, inter_d_block: int,
                 dropout: None | float):
        """
        Initializes the Expert layer with multiple experts.

        Args:
            num_experts (int): Number of experts in the MoE layer.
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        print('using Gated Experts')
        self.n_blocks = n_blocks
        dim = d_block // num_experts
        inter_dim = inter_d_block // num_experts
        # Create 3D weight tensors for experts (num_experts, dim, inter_dim) and (num_experts, inter_dim, dim)
        self.W1 = nn.ParameterList()
        self.W2 = nn.ParameterList()
        self.W3 = nn.ParameterList()
        for i in range(n_blocks + 1):  # one more for the output layer!
            w1 = torch.zeros(num_experts, d_first if i == 0 else dim, inter_dim)
            w1 = init_rsqrt_uniform_(w1, w1.shape[-2])
            self.W1.append(nn.Parameter(w1))

            w2 = torch.zeros(num_experts, inter_dim, d_out if i == n_blocks else dim)
            w2 = init_rsqrt_uniform_(w2, w2.shape[-2])
            self.W2.append(nn.Parameter(w2))

            w3 = torch.zeros(num_experts, d_first if i == 0 else dim, inter_dim)
            w3 = init_rsqrt_uniform_(w3, w3.shape[-2])
            self.W3.append(nn.Parameter(w3))
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multiple experts in parallel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_experts, dim).
        """
        for i in range(self.n_blocks + 1):  # one more for final layer
            # x = torch.einsum('...nd,...dh->...nh', x, self.Weights[i]) #
            if i == 0:
                x_expanded = x.unsqueeze(1).unsqueeze(1)
            else:
                x_expanded = x
            hidden = torch.matmul(x_expanded, self.W1[i])  # (batch, num_experts, 1, inter_dim)
            gate = torch.matmul(x_expanded, self.W3[i])
            hidden = F.silu(hidden) * gate  # (batch, num_experts, inter_dim)
            if (i == 0) or (i == self.n_blocks):
                x = torch.matmul(hidden, self.W2[i])  # (batch, num_experts, 1, dim)
            else:
                x = torch.matmul(hidden, self.W2[i]) + x  # residual connection
            if i < self.n_blocks:
                if self.dropout is not None:
                    x = self.dropout(x)
        x = x.squeeze()
        return x.unsqueeze(-1) if x.dim() == 2 else x  # Shape: (batch_size, num_experts, dim)

# 256, 10, 64;
# 10, 64, 64
