from ppo_config import PPOConfig
import gymnasium as gym
from actor_critic import ActorCritic
from rollout_buffer import RolloutBuffer
from ppo_trainer import PPOTrainer


def train():
    config = PPOConfig()

    # create environment
    env = gym.make(config.env_id)

    # get observation and action space dimensions from env
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"Observation space dimension: {obs_dim}")
    print(f"Action space dimension: {act_dim}")

    # create actor-critic model
    model = ActorCritic(obs_dim, act_dim).to(config.device)
    print('model created')

    # create rollout buffer
    buffer = RolloutBuffer(obs_dim, act_dim, config)
    print('buffer created')

    # create PPO trainer
    trainer = PPOTrainer(env, model, buffer, config)
    print('trainer created')

    # training loop
    print('starting training...')
    trainer.train(total_updates=config.total_updates)

    pass

if __name__ == "__main__":
    train()
