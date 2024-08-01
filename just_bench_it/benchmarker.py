import numpy as np
from tqdm import tqdm
import requests
import json
import time
from functools import wraps
from just_bench_it.envs import get_env, ENVS
from colorama import init, Fore, Style
import multiprocessing as mp
import psutil
import signal
import sys
import random
import warnings
import dill
import multiprocessing.reduction
from concurrent.futures import ProcessPoolExecutor

# 使用dill替换pickle
multiprocessing.reduction.ForkingPickler.dumps = dill.dumps
multiprocessing.reduction.ForkingPickler.loads = dill.loads

# 忽略 urllib3 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# 初始化 colorama
init()

GITHUB_REPO_OWNER = "justbechit"
GITHUB_REPO_NAME = "rl_ladder.github.io"
GITHUB_CLIENT_ID = "Ov23li6k9dJ9Ws9bsWy0"
GITHUB_CLIENT_SECRET = "5c8ab1c55c159ecb683d7fbbfe1efe657dc1d536"
MIN_EVAL_EPISODES = 20

# 定义一些颜色
COLORS = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

# ... [其他函数保持不变] ...

def benchmark(pretrained=False, train_episodes=1000, eval_episodes=100, comment=None, parallel_envs=1):
    def decorator(agent_class):
        @wraps(agent_class)
        def wrapper(*args, **kwargs):
            agent = agent_class(*args, **kwargs)
            agent.pretrained = pretrained  # 在这里设置 pretrained 属性

            def bench(self):
                nonlocal parallel_envs
                results = {}

                if eval_episodes < MIN_EVAL_EPISODES:
                    print(f"{Fore.YELLOW}Warning: The current number of evaluation episodes ({eval_episodes}) is too small for reliable results.")
                    print(f"Results can only be used locally and cannot be submitted for official evaluation.")
                    print(f"Minimum recommended evaluation episodes: {MIN_EVAL_EPISODES}{Style.RESET_ALL}")

                cpu_count = psutil.cpu_count(logical=False)
                if parallel_envs > cpu_count:
                    print(f"{Fore.YELLOW}Warning: You've set {parallel_envs} parallel environments, but your machine has {cpu_count} CPU cores.")
                    print(f"Consider reducing parallel_envs to {cpu_count} or less for optimal performance.{Style.RESET_ALL}")
                elif parallel_envs < cpu_count:
                    print(f"{Fore.GREEN}Tip: Your machine has {cpu_count} CPU cores. You might be able to increase parallel_envs to {cpu_count} for faster evaluation.{Style.RESET_ALL}")

                print(f"{Fore.CYAN}Running benchmark with {parallel_envs} parallel environments.{Style.RESET_ALL}")

                tasks = [(self, env_name, train_episodes, eval_episodes) for env_name in ENVS]

                with ProcessPoolExecutor(max_workers=parallel_envs) as executor:
                    futures = [executor.submit(process_env_wrapper, task) for task in tasks]
                    
                    with tqdm(total=len(tasks), desc="Total Progress", position=0) as pbar:
                        for future in futures:
                            result = future.result()
                            results[result[0]] = result[1]
                            pbar.update(1)

                print(f"\n{Fore.CYAN}{'=' * 40}")
                print(f"{Fore.YELLOW}Benchmark Results for {self.__class__.__name__}")
                print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")

                for env_name, score in results.items():
                    color = Fore.GREEN if score > 0 else Fore.RED
                    print(f"{color}{env_name:<20}: {score:.2f}{Style.RESET_ALL}")

                print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")

                if eval_episodes < MIN_EVAL_EPISODES:
                    return results

                upload = input(f"{Fore.YELLOW}Do you want to upload these results? (y/n): {Style.RESET_ALL}")
                if upload.lower() != 'y':
                    print("Results not uploaded.")
                    return results

                issue_title = f"Benchmark Results: {self.__class__.__name__}"
                issue_body = f"Algorithm: {self.__class__.__name__}\n"
                issue_body += "Benchmark results:\n\n"
                for env_name, score in results.items():
                    issue_body += f"- {env_name}: {score:.2f}\n"
                issue_body += f"\nPretrained: {pretrained}\n"
                issue_body += f"Train episodes: {train_episodes}\n"
                issue_body += f"Eval episodes: {eval_episodes}\n"
                if comment:
                    issue_body += f"\nComment: {comment}\n"

                create_github_issue(issue_title, issue_body, labels=["benchmark"])
                self.results = results
                return results

            agent.bench = bench.__get__(agent)
            return agent
        return wrapper
    return decorator

def process_env_wrapper(args):
    agent, env_name, train_episodes, eval_episodes = args
    return process_env(agent, env_name, train_episodes, eval_episodes)

def process_env(agent, env_name, train_episodes, eval_episodes):
    env = get_env(env_name)
    env_info = {
        'name': env_name,
        'action_space': env.action_space,
        'observation_space': env.observation_space,
        'env': env,
        'train_episodes': train_episodes,
        'eval_episodes': eval_episodes
    }
    agent.set_env_info(env_info)

    color = random.choice(COLORS)

    if not agent.pretrained:
        train_agent(agent, env, train_episodes, color, env_name)

    score = evaluate_agent(agent, env, eval_episodes, color, env_name)
    return env_name, float(score)

def train_agent(agent, env, episodes, color, env_name, max_steps=100):
    total_reward = 0
    with tqdm(total=episodes, desc=f"{color}Training {env_name}", unit="episode", leave=False, position=1) as pbar:
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            while not done and step < max_steps:
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                step += 1

            total_reward += episode_reward
            avg_reward = total_reward / (episode + 1)
            pbar.set_postfix({"avg_reward": f"{avg_reward:.2f}"})
            pbar.update(1)

def evaluate_agent(agent, env, episodes, color, env_name, max_steps=10000):
    scores = []
    with tqdm(total=episodes, desc=f"{color}Evaluating {env_name}", unit="episode", leave=False, position=1) as pbar:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            for _ in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                state = next_state
            scores.append(total_reward)
            pbar.set_postfix({"avg_score": f"{np.mean(scores):.2f}"})
            pbar.update(1)

    return np.mean(scores)

if __name__ == "__main__":
    # 示例用法
    @benchmark(pretrained=False, train_episodes=1000, eval_episodes=100, parallel_envs=4)
    class RandomAgent:
        def __init__(self):
            self.pretrained = False

        def set_env_info(self, env_info):
            self.env_info = env_info

        def act(self, state):
            return self.env_info['env'].action_space.sample()

        def update(self, state, action, reward, next_state, done):
            pass

    agent = RandomAgent()
    results = agent.bench()
    print(results)
