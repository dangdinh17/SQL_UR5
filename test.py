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
import pybullet as p

# Example usage
if __name__ == "__main__":
    render = True
    env = ur5GymEnv(renders=render)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SoftQLearning(state_dim, action_dim)
    checkpoint = 'saved_rl_models/checkpoint/ur5e_episode_950.pth'
    agent.load_checkpoint(checkpoint)
    total_reward = 0 
    for i in range(5):
        state = env.reset()
        for t in range(1, 120):
            print(t)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action, _ = agent.policy_network.sample_action(state_tensor)
            action = action.detach().cpu().numpy()[0]

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            state = next_state
    env.close()
        
   