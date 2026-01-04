import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(256, 4)
        self.log_std_linear = nn.Linear(256, 4)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        x = self.net(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        a_raw = normal.rsample()
        action = torch.tanh(a_raw)

        log_prob = normal.log_prob(a_raw) - torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob