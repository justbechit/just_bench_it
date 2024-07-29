import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from just_bench_it import benchmark, print_results
from just_bench_it import benchmark, run_benchmark, set_github_token
import torch
import torch.nn as nn
import torch.optim as optim

# 设置 GitHub token
set_github_token("your_github_token_here")


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

@benchmark(pretrained=False, train_episodes=1000, eval_episodes=100)
class DQNAgent:
    def __init__(self, learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, buffer_size=10000):
        self.env_info = None
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def set_env_info(self, env_info):
        self.env_info = env_info
        input_dim = env_info['observation_space'].shape[0]
        output_dim = env_info['action_space'].n
        self.q_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env_info['action_space'].sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        # Periodically update target network
        if random.random() < 0.01:  # 1% chance to update target network
            self.target_network.load_state_dict(self.q_network.state_dict())

# 运行基准测试
results = DQNAgent()
print_results(results)
