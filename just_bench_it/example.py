import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from just_bench_it import benchmark, print_results

class DQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # Calculate the size of flattened input
        flattened_size = int(np.prod(input_shape))
        
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
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
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_info = None
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.replay_buffer = None

    def set_env_info(self, env_info):
        self.env_info = env_info
        input_shape = env_info['observation_space'].shape
        output_dim = env_info['action_space'].n
        self.q_network = DQN(input_shape, output_dim).to(self.device)
        self.target_network = DQN(input_shape, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def act(self, state):
        if self.env_info is None:
            raise ValueError("Environment info not set. Call set_env_info first.")
        if random.random() < self.epsilon:
            return self.env_info['action_space'].sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        if self.replay_buffer is None:
            raise ValueError("Replay buffer not initialized. Call set_env_info first.")
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if random.random() < 0.01:  # 1% chance to update target network
            self.target_network.load_state_dict(self.q_network.state_dict())

# 运行基准测试
if __name__ == "__main__":
    agent = DQNAgent()
    results = agent()  # 这里调用 agent() 会触发 benchmark 装饰器
    print_results(results)
