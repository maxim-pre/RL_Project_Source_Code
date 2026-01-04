from dataclasses import dataclass, field
import torch


@dataclass
class PPOConfig:
    env_id: str = "BipedalWalker-v3"

    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    lr: float = 3e-4
    num_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 4096
    total_updates: int = 1000
    loss_coef: float = 0.5
    entropy_coef: float = 0.01
    log_std_min: int = -5
    log_std_max: int = 2

    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")) # enables dynamic default values
