import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import gymnasium as gym
import pickle
import time
from typing import Tuple, List
import os
from gymnasium.wrappers import RecordVideo


class DDPGAgent:

    def __init__(self,
                 critic_network,
                 actor_network,
                 device=torch.device("cpu"),
                 hardcore=False,
                 max_buffer_length=1_000_000):
        # Device.
        self.device = device

        # Environment.
        self.hardcore = hardcore
        self.env = gym.make("BipedalWalker-v3", hardcore=self.hardcore, render_mode=None)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_min = torch.tensor(self.env.action_space.low, device=self.device)
        self.action_max = torch.tensor(self.env.action_space.high, device=self.device)
        self.max_reward = self.env.spec.reward_threshold

        # Buffer.
        self.max_buffer_length = max_buffer_length
        self.buffer_write_idx = 0
        self.buffer_fullness = 0
        self.buffer_states = torch.zeros((self.max_buffer_length, self.state_size), dtype=torch.float32, device=self.device)
        self.buffer_actions = torch.zeros((self.max_buffer_length, self.action_size), dtype=torch.float32, device=self.device)
        self.buffer_rewards = torch.zeros((self.max_buffer_length, 1), dtype=torch.float32, device=self.device)
        self.buffer_next_states = torch.zeros((self.max_buffer_length, self.state_size), dtype=torch.float32, device=self.device)
        self.buffer_dones = torch.zeros((self.max_buffer_length, 1), dtype=torch.float32, device=self.device)

        # Networks.
        self.critic_network = copy.deepcopy(critic_network)
        for p in self.critic_network.parameters():
            p.requires_grad = True
        self.target_critic_network = copy.deepcopy(critic_network)
        for p in self.target_critic_network.parameters():
            p.requires_grad = False

        self.actor_network = copy.deepcopy(actor_network)
        for p in self.actor_network.parameters():
            p.requires_grad = True
        self.target_actor_network = copy.deepcopy(actor_network)
        for p in self.target_actor_network.parameters():
            p.requires_grad = False

        # Move networks to device.
        self.actor_network.to(self.device)
        self.target_actor_network.to(self.device)
        self.critic_network.to(self.device)
        self.target_critic_network.to(self.device)

        # Optimizers.
        self.critic_optimizer = None
        self.actor_optimizer = None

    def learn(self,
              n_episodes=2000,
              discount_factor=0.99,
              minibatch_size=256,
              tau=0.005,
              random_exploration_steps=0,
              actor_exploration_steps=0,
              vid_every=50,
              stop_after=None,
              reset_optim=True,
              reset_buffer=True,
              # ====================== #
              critic_lr=1e-3,
              actor_lr=1e-4,
              critic_grad_clip=1.0,
              actor_grad_clip=1.0,
              exploratory_noise_start=0.3,
              exploratory_noise_min=0.05,
              exploratory_noise_decay=2.5e-7,
              exploratory_noise_clip=0.3,
              updates_per_step=1):

        if reset_optim:
            self.reset_optim(critic_lr, actor_lr)
        if reset_buffer:
            self.reset_buffer()

        self.random_exploration(random_exploration_steps)
        self.actor_exploration(actor_exploration_steps, exploratory_noise_start, exploratory_noise_clip)

        total_step_count = 0
        episode_rewards = []
        episode_step_counts = []
        episode_run_times = []

        for n in range(n_episodes):

            print(f"Running Episode {n + 1}...")
            start_time = time.time()
            state, _ = self.env.reset()

            done = False
            episode_reward = 0
            episode_step_count = 0

            while not done:

                exploratory_noise = max(exploratory_noise_min,
                                        exploratory_noise_start - exploratory_noise_decay * total_step_count)

                action = self.select_action(state,
                                            add_noise=True,
                                            exploratory_noise=exploratory_noise,
                                            exploratory_noise_clip=exploratory_noise_clip)
                new_state, reward, terminal, truncated, _ = self.env.step(action)
                done = terminal or truncated

                self.save_transition(state, action, reward, new_state, done)

                episode_step_count += 1
                total_step_count += 1
                episode_reward += reward

                if self.buffer_fullness >= minibatch_size:

                    for _ in range(updates_per_step):

                        minibatch = self.sample_minibatch(minibatch_size)
                        self.update_critic_network(minibatch, discount_factor, critic_grad_clip)
                        self.update_actor_network(minibatch, actor_grad_clip)
                        self.soft_update_target_critic(tau)
                        self.soft_update_target_actor(tau)

                state = new_state

            end_time = time.time()
            episode_run_time = end_time - start_time

            print(f"Reward: {episode_reward:.2f} - Step Count: {episode_step_count} - Run Time: {episode_run_time:.2f}s - Total Step Count: {total_step_count} - Exploratory Noise: {exploratory_noise:.5f}")
            episode_rewards.append(episode_reward)
            episode_step_counts.append(episode_step_count)
            episode_run_times.append(episode_run_time)

            if stop_after is not None and all(ep_rew >= self.max_reward for ep_rew in episode_rewards[-stop_after:]):
                break

            if n % vid_every == 0:
                self.test_episode(video=True, video_name=f"episode-{n + 1}")

        self.save_run(episode_rewards,
                      episode_step_counts,
                      episode_run_times,
                      n_episodes,
                      discount_factor,
                      minibatch_size,
                      tau,
                      random_exploration_steps,
                      actor_exploration_steps,
                      stop_after,
                      reset_optim,
                      reset_buffer,
                      critic_lr,
                      actor_lr,
                      critic_grad_clip,
                      actor_grad_clip,
                      exploratory_noise_start,
                      exploratory_noise_min,
                      exploratory_noise_decay,
                      exploratory_noise_clip,
                      updates_per_step)

    def reset_optim(self, critic_lr, actor_lr):
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_lr)

    def reset_buffer(self):
        self.buffer_write_idx = 0
        self.buffer_fullness = 0
        self.buffer_states = torch.zeros((self.max_buffer_length, self.state_size), dtype=torch.float32, device=self.device)
        self.buffer_actions = torch.zeros((self.max_buffer_length, self.action_size), dtype=torch.float32, device=self.device)
        self.buffer_rewards = torch.zeros((self.max_buffer_length, 1), dtype=torch.float32, device=self.device)
        self.buffer_next_states = torch.zeros((self.max_buffer_length, self.state_size), dtype=torch.float32, device=self.device)
        self.buffer_dones = torch.zeros((self.max_buffer_length, 1), dtype=torch.float32, device=self.device)

    def random_exploration(self, num_steps: int):
        print("Performing Random Exploration...")
        step_count = 0
        state, _ = self.env.reset()

        while step_count < num_steps:

            action_tensor = self.action_min + (self.action_max - self.action_min) * torch.rand(self.action_size, device=self.device)
            action = action_tensor.cpu().numpy()
            new_state, reward, terminal, truncated, _ = self.env.step(action)
            done = terminal or truncated
            self.save_transition(state, action, reward, new_state, done)

            step_count += 1
            state = new_state

            if done:
                state, _ = self.env.reset()

    def actor_exploration(self, num_steps: int, exploratory_noise: float, exploratory_noise_clip: float):
        print("Performing Actor Exploration...")
        step_count = 0
        state, _ = self.env.reset()

        while step_count < num_steps:

            action = self.select_action(state,
                                        add_noise=True,
                                        exploratory_noise=exploratory_noise,
                                        exploratory_noise_clip=exploratory_noise_clip)
            new_state, reward, terminal, truncated, _ = self.env.step(action)
            done = terminal or truncated
            self.save_transition(state, action, reward, new_state, done)

            step_count += 1
            state = new_state

            if done:
                state, _ = self.env.reset()

    def select_action(self,
                      state: np.ndarray,
                      add_noise: bool=False,
                      exploratory_noise: float=None,
                      exploratory_noise_clip: float=None) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor = self.actor_network(state_tensor).squeeze(0)
            if add_noise:
                noise = torch.randn(self.action_size, device=self.device) * exploratory_noise
                noise_clipped = torch.clamp(noise, min=-exploratory_noise_clip, max=exploratory_noise_clip)
                action_tensor = action_tensor + noise_clipped
            action_tensor = torch.clamp(action_tensor, min=self.action_min, max=self.action_max)
            return action_tensor.cpu().numpy()

    def save_transition(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
        self.buffer_states[self.buffer_write_idx] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.buffer_actions[self.buffer_write_idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.buffer_rewards[self.buffer_write_idx] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.buffer_next_states[self.buffer_write_idx] = torch.tensor(new_state, dtype=torch.float32, device=self.device)
        self.buffer_dones[self.buffer_write_idx] = torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=self.device)

        self.buffer_write_idx = (self.buffer_write_idx + 1) % self.max_buffer_length
        self.buffer_fullness = min(self.buffer_fullness + 1, self.max_buffer_length)

    def sample_minibatch(self, minibatch_size: int) -> Tuple[torch.tensor, ...]:
        indices = torch.randint(0, self.buffer_fullness, (minibatch_size,), device=self.device)

        mb_states = self.buffer_states[indices]
        mb_actions = self.buffer_actions[indices]
        mb_rewards = self.buffer_rewards[indices]
        mb_next_states = self.buffer_next_states[indices]
        mb_dones = self.buffer_dones[indices]

        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones

    def update_critic_network(self,
                               minibatch: Tuple[torch.tensor, ...],
                               discount_factor: float,
                               critic_grad_clip: float):
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = minibatch
        mb_state_actions = torch.cat([mb_states, mb_actions], dim=1)

        with torch.no_grad():
            next_actions = self.target_actor_network(mb_next_states)
            next_state_actions = torch.cat((mb_next_states, next_actions), dim=1)
            q_next = self.target_critic_network(next_state_actions)
            q_target = mb_rewards + discount_factor * (1.0 - mb_dones) * q_next

        q_expected = self.critic_network(mb_state_actions)
        critic_loss = torch.mean((q_target - q_expected) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if critic_grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.critic_network.parameters(), critic_grad_clip)
        self.critic_optimizer.step()

    def update_actor_network(self,
                             minibatch: Tuple[torch.tensor, ...],
                             actor_grad_clip: float):
        mb_states, *_ = minibatch

        for p in self.critic_network.parameters():
            p.requires_grad = False

        raw_actions = self.actor_network(mb_states)
        raw_state_actions = torch.cat((mb_states, raw_actions), dim=1)
        actor_loss = -self.critic_network(raw_state_actions).mean()

        for p in self.critic_network.parameters():
            p.requires_grad = True

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.actor_network.parameters(), actor_grad_clip)
        self.actor_optimizer.step()

    def soft_update_target_critic(self, tau: float):
        with torch.no_grad():
            for w_target, w_local in zip(self.target_critic_network.parameters(), self.critic_network.parameters()):
                w_target.data.copy_(tau * w_local.data + (1 - tau) * w_target.data)

    def soft_update_target_actor(self, tau: float):
        with torch.no_grad():
            for w_target, w_local in zip(self.target_actor_network.parameters(), self.actor_network.parameters()):
                w_target.data.copy_(tau * w_local.data + (1 - tau) * w_target.data)

    def test_episode(self, video: bool=False, video_name: str=None):
        print("\n========TEST RUN========")

        if video:
            test_env = gym.make("BipedalWalker-v3", hardcore=self.hardcore, render_mode="rgb_array")
            test_env = RecordVideo(test_env, video_folder="videos", episode_trigger=lambda x: True)
        else:
            test_env = gym.make("BipedalWalker-v3", hardcore=self.hardcore, render_mode="human")

        s, _ = test_env.reset()
        test_episode_reward = 0
        test_episode_step_count = 0
        test_start_time = time.time()

        while True:

            a = self.select_action(s, add_noise=False, exploratory_noise=None, exploratory_noise_clip=None)

            s_, r, terminal, truncated, _ = test_env.step(a)
            done = terminal or truncated

            test_episode_reward += r
            test_episode_step_count += 1

            if done:
                break

            s = s_

        test_episode_end_time = time.time()
        test_episode_run_time = test_episode_end_time - test_start_time
        test_env.close()

        if video:
            os.rename("videos/rl-video-episode-0.mp4", f"videos/{video_name}.mp4")

        print(f"Reward: {test_episode_reward:.2f} - Step Count: {test_episode_step_count} - Run Time: {test_episode_run_time:.2f}s\n")

    def save_run(self,
                 episode_rewards: List,
                 episode_step_counts: List,
                 episode_run_times: List,
                 n_episodes: int,
                 discount_factor: float,
                 minibatch_size: int,
                 tau: float,
                 random_exploration_steps: int,
                 actor_exploration_steps: int,
                 stop_after: int,
                 reset_optim: bool,
                 reset_buffer: bool,
                 critic_lr: float,
                 actor_lr: float,
                 critic_grad_clip: float,
                 actor_grad_clip: float,
                 exploratory_noise_start: float,
                 exploratory_noise_min: float,
                 exploratory_noise_decay: float,
                 exploratory_noise_clip: float,
                 updates_per_step: int):

        torch.save(self.critic_network.state_dict(), "outputs/critic_network1.pth")
        torch.save(self.actor_network.state_dict(), "outputs/actor_network.pth")

        with open("outputs/episode_rewards.pkl", "wb") as f:
            pickle.dump(episode_rewards, f)
        with open("outputs/episode_step_counts.pkl", "wb") as f:
            pickle.dump(episode_step_counts, f)
        with open("outputs/episode_run_times.pkl", "wb") as f:
            pickle.dump(episode_run_times, f)

        settings = {
            "hardcore": self.hardcore,
            "max_buffer_length": self.max_buffer_length,
            "n_episodes": n_episodes,
            "discount_factor": discount_factor,
            "minibatch_size": minibatch_size,
            "tau": tau,
            "random_exploration_steps": random_exploration_steps,
            "actor_exploration_steps": actor_exploration_steps,
            "stop_after": stop_after,
            "reset_optim": reset_optim,
            "reset_buffer": reset_buffer,
            "critic_lr": critic_lr,
            "actor_lr": actor_lr,
            "critic_grad_clip": critic_grad_clip,
            "actor_grad_clip": actor_grad_clip,
            "exploratory_noise_start": exploratory_noise_start,
            "exploratory_noise_min": exploratory_noise_min,
            "exploratory_noise_decay": exploratory_noise_decay,
            "exploratory_noise_clip": exploratory_noise_clip,
            "updates_per_step": updates_per_step
        }

        with open("outputs/settings.pkl", "wb") as f:
            pickle.dump(settings, f)
