import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
'''
training loop:
1. collect rollout data using current policy
2. compute last value for GAE 
3. compute advantages and returns
4. get mini-batches from rollout buffer
5. apply PPO update for each mini-batch
6. clear rollout buffer
7. repeat for num_updates
'''

class PPOTrainer:
    def __init__(self, env, model, buffer, config):
        self.env = env
        self.model = model # actor-critic model
        self.buffer = buffer
        self.config = config

        self.writer = SummaryWriter("runs/ppo_bipedalwalker")

        # hyperparameters from config 
        self.gamma = config.gamma
        self.lam = config.lam
        self.clip_epsilon = config.clip_epsilon
        self.lr = config.lr
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.device = config.device
        self.loss_coef = config.loss_coef
        self.entropy_coef = config.entropy_coef

        # used for logging
        self.episode_rewards = []
        self.current_episode_reward = 0

        self.optimizer = Adam(self.model.parameters(), lr=config.lr)


    def collect_rollout(self):
        # reset environment and get initial observation
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device) 

        # set rollout buffer pointer to 0
        self.buffer.pointer = 0

        while self.buffer.pointer < self.config.buffer_size:
            action, log_prob, value = self.model.get_action(obs)

            #apply action in environment
            next_obs, reward, done, truncated, _ = self.env.step(action.cpu().numpy())

            # accumulate episode reward for logging
            self.current_episode_reward += reward

            # store timestep in rollout buffer
            self.buffer.add(
                obs=obs, 
                action=action, 
                log_prob=log_prob, 
                reward=torch.tensor(reward, device=self.device),
                done=torch.tensor(float(done and not truncated), device=self.device),
                value=value 
            )

            # handle episode termination
            if done or truncated:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                next_obs, _ = self.env.reset() # if episode ends, next_obs becomes first state of new epiode

            
            # move to next observation
            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        
        # Still need to compute value for last observation
        with torch.no_grad():
            if self.buffer.dones[self.buffer.pointer -1] > 0.5:
                last_value = torch.tensor(0.0, device=self.device)
            else:
                last_value = self.model.get_value(obs)
        
        # compute and store advantages and returns in rollout buffer
        self.buffer.compute_advantages_and_returns(last_value, self.gamma, self.lam)
    

    def ppo_update(self):
        
        # reuse rollout data for multiple epochs
        for _ in range(self.num_epochs):

            for obs, actions, old_log_probs, advantages, returns, old_values in self.buffer.get_batches(self.batch_size):

                log_probs, entropy, values = self.model.evaluate_actions(obs, actions)

                # compute ratio to measure how much the policy has changed from the old one
                ratio = torch.exp(log_probs - old_log_probs)

                unclipped_obj = ratio * advantages
                clipped_obj = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

                value_pred_clipped = old_values + torch.clamp(values - old_values, -self.clip_epsilon, self.clip_epsilon)
                value_loss_unclipped = (values - returns).pow(2)
                value_loss_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_bonus = entropy.mean()
                loss = policy_loss + self.loss_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
    
    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"model saved to {path}")
    

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"model loaded from {path}")
    
    def train(self, total_updates=1000):
        for update in range(total_updates):
            self.collect_rollout()
            self.ppo_update()

            if len(self.episode_rewards) > 0:
                avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10)
                print(f"Update {update + 1}/{total_updates}, Average Reward (last 10 episodes): {avg_reward:.2f}")
                self.writer.add_scalar("Average Reward", avg_reward, update)
        
        self.save("ppo_bipedalwalker.pt")


    def evaluate(self, num_episodes=5):
        self.model.eval()
        env = gym.make(self.config.env_id, render_mode="human")


        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False 
            truncated = False
            total_reward = 0 

            while not (done or truncated):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                mu, std, value = self.model.forward(obs_tensor)
                action = mu.detach().cpu().numpy()  # take mean action for evaluation

                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward 

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")
                    


        