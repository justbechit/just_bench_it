import random
from just_bench_it import benchmark

@benchmark(pretrained=False, train_episodes=1, eval_episodes=1, comment="Random Agent benchmark")
class RandomAgent:
    def __init__(self):
        self.env_info = None

    def set_env_info(self, env_info):
        self.env_info = env_info

    def act(self, state):
        return self.env_info['action_space'].sample()

    def update(self, state, action, reward, next_state, done):
        # 随机agent不需要更新
        pass

if __name__ == "__main__":
    agent = RandomAgent()
    results = agent.bench()
    print(results)
