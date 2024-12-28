from torch import nn
import torch

from torch.distributions import Normal


# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0, device='cuda'):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        # Variational parameters for posterior
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features)).to(self.device)
        self.weight_logvar = nn.Parameter(torch.zeros(out_features, in_features)).to(self.device)
        self.bias_mu = nn.Parameter(torch.zeros(out_features)).to(self.device)
        self.bias_logvar = nn.Parameter(torch.zeros(out_features)).to(self.device)

        # Prior distributions
        self.prior_weight = Normal(torch.zeros_like(self.weight_mu), prior_std * torch.ones_like(self.weight_mu))
        self.prior_bias = Normal(torch.zeros_like(self.bias_mu), prior_std * torch.ones_like(self.bias_mu))

    def forward(self, x):
        # Reparameterization trick to sample weights and biases
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)

        # Linear transformation
        return torch.addmm(bias, x, weight.t())

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

    def forward(self, x):
        logits = self.blin(x)  # shape: (batch, num_experts)
        alpha = torch.softmax(logits, dim=1)  # gating coefficients
        return alpha

    def kl_loss(self):
        """
        Sum KL from each BayesianLinear sub-layer.
        """
        # return self.blin1.kl_loss() + self.blin2.kl_loss()
        return self.blin.kl_loss()
