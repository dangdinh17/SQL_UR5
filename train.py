import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from collections import deque
from env import ur5GymEnv
from matplotlib import pyplot as plt
from tqdm import tqdm
from sql import *

# Example usage
if __name__ == "__main__":
    render = True
    env = ur5GymEnv(renders=render)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SoftQLearning(state_dim, action_dim)

    num_episodes = 1000
    batch_size = 64

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in tqdm(range(100), desc=f"Episode {episode + 1} Steps"):
            state = agent.normalize_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action, _ = agent.policy_network.sample_action(state_tensor)
            action = action.detach().cpu().numpy()[0]

            next_state, reward, done, _ = env.step(action)
            agent.add_experience(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            if len(agent.replay_buffer.buffer) > batch_size:
                agent.update(batch_size)

            if done:
                print("Done...")
                agent.save_checkpoint(filepath=f'saved_rl_models/ur5e_best_{num_episodes}.pth')
                break

        agent.episode_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if episode % 50 == 0:
            agent.save_checkpoint(filepath=f'saved_rl_models/checkpoint/ur5e_episode_{episode}.pth')
    env.close()
    # Plot results
    plt.figure(figsize=(12, 4))

    # Episode Rewards
    plt.subplot(1, 3, 1)
    plt.plot(agent.episode_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")
    plt.legend()

    # Q Loss
    plt.subplot(1, 3, 2)
    plt.plot(agent.q_losses, label="Q Loss")
    plt.xlabel("Update Steps")
    plt.ylabel("Loss")
    plt.title("Q Loss")
    plt.legend()

    # Policy Loss
    plt.subplot(1, 3, 3)
    plt.plot(agent.policy_losses, label="Policy Loss")
    plt.xlabel("Update Steps")
    plt.ylabel("Loss")
    plt.title("Policy Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"saved_rl_models/training_result_{np.random.randint(0, 100)}.png", dpi=300)
    print("Training plots saved...")

    plt.show()