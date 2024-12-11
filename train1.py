import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from collections import deque
from env import ur5GymEnv
from matplotlib import pyplot as plt
from tqdm import tqdm

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(-1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(-1)
        )

# Soft Q-Learning implementation
class SoftQLearning:
    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99, tau=0.005, alpha=0.5, k=1e-4, buffer_size=100000):
        self.gamma = gamma
        self.tau = tau
        self.alpha_0 = alpha
        self.alpha = alpha
        self.k = k  # Decay rate for alpha

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.q_losses = []
        self.policy_losses = []
        self.episode_rewards = []

        self.update_step = 0

    def update(self, batch_size):
        # Decay alpha
        self.alpha = self.alpha_0 * np.exp(-self.k * self.update_step)
        self.update_step += 1

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Update Q-function
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.sample_action(next_states)
            target_q = self.target_q_network(next_states, next_actions)
            target_value = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_probs)

        q_values = self.q_network(states, actions)
        q_loss = ((q_values - target_value) ** 2).mean()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        new_actions, log_probs = self.policy_network.sample_action(states)
        q_new_actions = self.q_network(states, new_actions)
        policy_loss = (self.alpha * log_probs - q_new_actions).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.q_losses.append(q_loss.item())

        # Update target Q-network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def normalize_state(self, state):
        return (state - np.mean(state)) / (np.std(state) + 1e-6)

    def save_checkpoint(self, filepath):
        """Lưu checkpoint bao gồm trạng thái mô hình, optimizer và các giá trị cần thiết khác."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Tải checkpoint và khôi phục trạng thái mô hình."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")

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

        for step in tqdm(range(200), desc=f"Episode {episode + 1} Steps"):
            state = agent.normalize_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = agent.policy_network.sample_action(state_tensor)
            action = action.detach().numpy()[0]

            next_state, reward, done, _ = env.step(action)
            agent.add_experience(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            if len(agent.replay_buffer.buffer) > batch_size:
                agent.update(batch_size)

            if done:
                print("Done...")
                agent.save_checkpoint(filepath=f'saved_rl_models/ur5e_best.pth')
                break

        agent.episode_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if episode % 100 == 0:
            agent.save_checkpoint(filepath=f'saved_rl_models/checkpoint/ur5e_episode_{episode}.pth')

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
    plt.savefig("saved_rl_models/training_result.png", dpi=300)  # Bạn có thể đổi tên và định dạng tệp nếu muốn
    print("Biểu đồ đã được lưu thành công...")

    plt.show()
