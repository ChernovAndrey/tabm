from torch import nn
import torch

from torch.distributions import Normal
from .util import init_rsqrt_uniform_
import math


def truncated_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Apply truncated normal initialization to a tensor.

    Parameters:
        tensor (torch.Tensor): The tensor to initialize.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        a (float): Lower truncation boundary.
        b (float): Upper truncation boundary.
    """
    with torch.no_grad():
        # Fill the tensor with a normal distribution
        tensor.normal_(mean=mean, std=std)
        # Clamp values to lie within [a, b]
        tensor.clamp_(min=a, max=b)


# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0, device='cuda'):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        # Variational parameters for posterior

        # Initialize weight_mu and weight_logvar
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, device=self.device))
        self.weight_logvar = nn.Parameter(torch.empty(out_features, in_features, device=self.device))

        # Initialize bias_mu and bias_logvar
        self.bias_mu = nn.Parameter(torch.empty(out_features, device=self.device))
        self.bias_logvar = nn.Parameter(torch.empty(out_features, device=self.device))

        truncated_normal_(self.weight_mu, mean=0.0, std=self.prior_std, a=-2 * self.prior_std, b=2 * prior_std)
        # init_rsqrt_uniform_(self.weight_mu, self.weight_mu.shape[-1])
        # nn.init.xavier_normal_(self.weight_mu)
        # nn.init.normal_(self.weight_mu)
        # nn.init.constant_(self.weight_mu, 0.0)
        nn.init.constant_(self.bias_mu, 0.0)

        # nn.init.constant_(self.weight_logvar, math.log(self.prior_std**2))
        log_var_mean_prior = math.log(self.prior_std ** 2)
        log_var_std_prior = self.prior_std / 10

        truncated_normal_(self.weight_logvar, mean=log_var_mean_prior, std=log_var_std_prior,
                          a=log_var_mean_prior - 2.0 * log_var_std_prior,
                          b=log_var_mean_prior + 2.0 * log_var_std_prior)

        nn.init.constant_(self.bias_logvar, log_var_mean_prior)
        # Prior distributions
        self.prior_weight = Normal(torch.zeros_like(self.weight_mu), prior_std * torch.ones_like(self.weight_mu))
        self.prior_bias = Normal(torch.zeros_like(self.bias_mu), prior_std * torch.ones_like(self.bias_mu))

    # def forward(self, x):
    #     # Reparameterization trick to sample weights and biases
    #     weight_std = torch.exp(0.5 * self.weight_logvar)
    #     bias_std = torch.exp(0.5 * self.bias_logvar)
    #     weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
    #     bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
    #
    #     # Linear transformation
    #     return torch.addmm(bias, x, weight.t())
    def forward(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): inout tensor
            num_samples (int): when num_samples = 0, it means that we take MAP estimation
            return_average (bool): whether to return mean or all samples

       Output:
            [B, out_dim] - when num_samples <= 1 or return_average=True
            [N, B, out_dim] - otherwise

        """
        if num_samples == 1:
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)

            # return torch.addmm(bias, x, weight.t())
            return nn.functional.linear(x, weight, bias)

        elif num_samples == 0:  # e.g MAP estimation
            # return torch.addmm(self.bias_mu, x, self.weight_mu.t())
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)

        weight_std = torch.exp(0.5 * self.weight_logvar).unsqueeze(0)
        bias_std = torch.exp(0.5 * self.bias_logvar).unsqueeze(0)

        weight_samples = self.weight_mu.unsqueeze(0) + weight_std * torch.randn(num_samples, *self.weight_mu.shape)

        # [samples, out_dim]
        bias_samples = self.bias_mu.unsqueeze(0) + bias_std * torch.randn(num_samples, *self.bias_mu.shape)

        # TODO: measure which approach (einsum or bmm) performs better
        # outputs = torch.einsum('noi,bi->nbo', weight_samples, x) + bias_samples.unsqueeze(1)

        # Make x into shape [N, B, in_dim]
        x_expanded = x.unsqueeze(0).expand(num_samples, -1, -1)  # [N, B, in_dim]

        # For matrix multiplication, we want [N, B, out_dim].
        # But weight_samples is [N, out_dim, in_dim], so we transpose last two dims:
        weight_t = weight_samples.transpose(1, 2)  # [N, in_dim, out_dim]

        outputs = torch.bmm(x_expanded, weight_t) + bias_samples.unsqueeze(1)  # [N, B, out_dim]

        # if return_average:
        #     outputs = outputs.mean(dim=0)

        return outputs

    # def kl_divergence(self):
    def kl_loss(self):
        # KL divergence between posterior and prior
        weight_posterior = Normal(self.weight_mu, torch.exp(0.5 * self.weight_logvar))
        bias_posterior = Normal(self.bias_mu, torch.exp(0.5 * self.bias_logvar))

        kl_weight = torch.distributions.kl.kl_divergence(weight_posterior, self.prior_weight).sum()
        kl_bias = torch.distributions.kl.kl_divergence(bias_posterior, self.prior_bias).sum()
        return kl_weight + kl_bias


###############################################################################
# 2. Bayesian Gating Network (two layers) with KL aggregator
###############################################################################
class BayesianGatingNetwork(nn.Module):
    """
    MLP gating with BayesianLinear for each layer.
    We'll provide a method to sum the KL from both layers.
    """

    def __init__(self, in_features=784, num_experts=3, prior_std=1.0, device='cuda'):
        super().__init__()
        self.blin = BayesianLinear(in_features, num_experts, prior_std=prior_std, device=device)

    def forward(self, x, num_samples: int):
        logits = self.blin(x, num_samples)  # shape: (batch, num_experts) or (num_samples, batch, num_experts)
        alpha = torch.softmax(logits, dim=-1)  # gating coefficients
        return alpha

    def kl_loss(self):
        """
        Sum KL from each BayesianLinear sub-layer.
        """
        # return self.blin1.kl_loss() + self.blin2.kl_loss()
        return self.blin.kl_loss()
