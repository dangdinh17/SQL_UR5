import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from env import *
from torch.distributions.categorical import Categorical

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, alpha):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, num_actions)
        self.alpha = alpha

    def get_Q(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_V(self, q):
        # print(q)
        # print(q.shape)
        v = self.alpha * torch.log(torch.mean(torch.exp(q / self.alpha), dim=-1))
        return v


def train_soft_q_learning(env, num_episodes=1000, batch_size=64, gamma=0.99, 
                           alpha=0.2, learning_rate=1e-3, buffer_capacity=10000):
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # Initialize networks
    q_network = SoftQNetwork(state_dim, num_actions, alpha)
    target_q_network = SoftQNetwork(state_dim, num_actions, alpha)
    target_q_network.load_state_dict(q_network.state_dict())

    # Optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Training loop
    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        done = False

        while not done:
            # Select action
            with torch.no_grad():
                q_values = q_network.get_Q(state)
                action = q_values.numpy()  # Use Q-values directly as action
                action = np.clip(action, env.action_space.low, env.action_space.high)

            # Step environment
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            
            # Store transition
            replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)
            state = next_state
            total_reward += reward

            # Learn from replay buffer
            if len(replay_buffer) >= batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states = torch.FloatTensor(states)
                next_states = torch.FloatTensor(next_states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Compute current Q values
                current_q_values = q_network.get_Q(states)
                
                # Compute target Q values
                with torch.no_grad():
                    next_q_values = target_q_network.get_Q(next_states)
                    next_v_value = target_q_network.get_V(next_q_values).unsqueeze(1)
                
                # Compute target
                target = rewards + gamma * (1 - dones) * next_v_value
                
                # Soft Q-learning loss
                q_loss = F.mse_loss(current_q_values, target)
                
                # Optimize
                optimizer.zero_grad()
                q_loss.backward()
                optimizer.step()

        # Soft update of target network
        for target_param, param in zip(target_q_network.parameters(), q_network.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

        episode_rewards.append(total_reward)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return episode_rewards, q_network

def main():
    # Create environment
    env = ur5GymEnv(renders=False)  # Set renders=True to visualize training
    
    # Train agent
    rewards, trained_model = train_soft_q_learning(env)
    
    # Plot rewards
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Save model
    torch.save(trained_model.state_dict(), 'ur5e_soft_q_model.pth')

if __name__ == "__main__":
    main()