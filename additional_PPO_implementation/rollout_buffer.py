import torch 

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, config):

        self.device = config.device
        self.buffer_size = config.buffer_size # number of timesteps before PPO update

        # allocating space for rollout data
        self.obs = torch.zeros((config.buffer_size, obs_dim), device=config.device)
        self.actions = torch.zeros((config.buffer_size, act_dim), device=config.device)
        self.log_probs = torch.zeros(config.buffer_size, device=config.device)
        self.rewards = torch.zeros(config.buffer_size, device=config.device)
        self.dones = torch.zeros(config.buffer_size, device=config.device)
        self.values = torch.zeros(config.buffer_size, device=config.device)

        # these get filled after rollout is complete
        self.advantages = torch.zeros(config.buffer_size, device=config.device)
        self.returns = torch.zeros(config.buffer_size, device=config.device)

        # pointer to current position in buffer 
        self.pointer = 0 
    
    def add(self, obs, action, log_prob, reward, done, value):
        # add a single timestep of data to the buffer and increment pointer
        self.obs[self.pointer] = obs
        self.actions[self.pointer] = action
        self.log_probs[self.pointer] = log_prob
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.values[self.pointer] = value

        self.pointer += 1
    
    def compute_advantages_and_returns(self, last_value, gamma=0.99, lam=0.95):
        # computes generalised advantage estimation (GAE) and returns after rollout is complete

        gae = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1: 
                next_value = last_value
            else:
                next_value = self.values[step + 1]

            next_non_terminal = 1.0 - self.dones[step]
            td_error = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]

            gae = td_error + gamma * lam * next_non_terminal * gae

            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]

        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)  # normalize advantages
    
    def get_batches(self, batch_size):
        # generate randomised mini-batches for the PPO updates

        indices = torch.randperm(self.buffer_size) # ramdomize to reduce policy overfitting

        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (self.obs[batch_indices],
                   self.actions[batch_indices],
                   self.log_probs[batch_indices],
                   self.advantages[batch_indices],
                   self.returns[batch_indices],
                   self.values[batch_indices])

