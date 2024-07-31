import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from just_bench_it import benchmark

class CNNMLP(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(CNNMLP, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # 假设输入是(n_channels, height, width)格式
        n_input_channels = input_shape[0]
        
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 计算卷积层输出的特征数量
        conv_out_size = self._get_conv_out_size(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _get_conv_out_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.device = device

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)).to(self.device),
                torch.LongTensor(action).to(self.device),
                torch.FloatTensor(reward).to(self.device),
                torch.FloatTensor(np.array(next_state)).to(self.device),
                torch.FloatTensor(done).to(self.device))

    def __len__(self):
        return len(self.buffer)

@benchmark(pretrained=False, train_episodes=1, eval_episodes=1, comment="CNN+MLP DQN benchmark")
class DQNAgent:
    def __init__(self, learning_rate=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, buffer_size=100000):
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

        # 对于ALE环境，可能需要调整输入shape
        if len(input_shape) == 2:  # 如果是2D输入
            input_shape = (1, *input_shape)  # 添加一个通道维度
        elif len(input_shape) == 3 and input_shape[0] == 1:  # 如果已经是3D但只有1个通道
            input_shape = (1, input_shape[1], input_shape[2])  # 保持1个通道

        self.q_network = CNNMLP(input_shape, output_dim).to(self.device)
        self.target_network = CNNMLP(input_shape, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.device)

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

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if random.random() < 0.01:  # 1% chance to update target network
            self.target_network.load_state_dict(self.q_network.state_dict())

if __name__ == "__main__":
    agent = DQNAgent()
    results = agent.bench()
    print(results)