import numpy as np
from tqdm import tqdm
import requests
import json
import time
from functools import wraps
from just_bench_it.envs import get_env, ENVS
from colorama import init, Fore, Style
import psutil
import sys
import random
import warnings

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

def get_device_code():
    url = "https://github.com/login/device/code"
    headers = {"Accept": "application/json"}
    data = {
        "client_id": GITHUB_CLIENT_ID,
        "scope": "repo"
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to obtain device code: {response.content}")
        return None

def poll_for_token(device_code, interval):
    url = "https://github.com/login/oauth/access_token"
    headers = {"Accept": "application/json"}
    data = {
        "client_id": GITHUB_CLIENT_ID,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
    }
    while True:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            response_data = response.json()
            if "access_token" in response_data:
                return response_data.get("access_token")
            elif "error" in response_data and response_data["error"] == "authorization_pending":
                print("Authorization pending. Waiting for user to authorize...")
                time.sleep(interval)
            else:
                print(f"Error in response: {response_data}")
                break
        else:
            print(f"Failed to poll for token: {response.content}")
            break
    return None

def get_github_token():
    device_data = get_device_code()
    if not device_data:
        return None
    print(f"Please go to {device_data['verification_uri']} and enter the code: {device_data['user_code']}")
    return poll_for_token(device_data["device_code"], device_data["interval"])

def create_github_issue(title, body, labels=None):
    github_token = get_github_token()
    if not github_token:
        print("Failed to obtain GitHub token. Skipping issue creation.")
        return None
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body,
        "labels": labels or []
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 201:
        print("Issue created successfully")
        return response.json()
    else:
        print(f"Failed to create issue: {response.content}")
        return None

def benchmark(pretrained=False, train_episodes=1000, eval_episodes=100, comment=None):
    def decorator(agent_class):
        @wraps(agent_class)
        def wrapper(*args, **kwargs):
            agent = agent_class(*args, **kwargs)
            agent.pretrained = pretrained

            def bench(self):
                results = {}

                if eval_episodes < MIN_EVAL_EPISODES:
                    print(f"{Fore.YELLOW}Warning: The current number of evaluation episodes ({eval_episodes}) is too small for reliable results.")
                    print(f"Results can only be used locally and cannot be submitted for official evaluation.")
                    print(f"Minimum recommended evaluation episodes: {MIN_EVAL_EPISODES}{Style.RESET_ALL}")

                print(f"{Fore.CYAN}Running benchmark{Style.RESET_ALL}")

                with tqdm(total=len(ENVS), desc="Total Progress", position=0) as pbar:
                    for env_name in ENVS:
                        env_result = process_env(self, env_name, train_episodes, eval_episodes)
                        results[env_result[0]] = env_result[1]
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

def process_env(agent, env_name, train_episodes, eval_episodes):
    # 设置环境信息
    env = get_env(env_name)
    env_info = {
        'name': env_name,
        'action_space': env.action_space,
        'observation_space': env.observation_space,
        'env': env
    }
    agent.set_env_info(env_info)

    # 训练和评估
    color = random.choice(COLORS)
    if not agent.pretrained:
        train_agent(agent, env, train_episodes, color, env_name)
    score = evaluate_agent(agent, env, eval_episodes, color, env_name)

    return env_name, score

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
    @benchmark(pretrained=False, train_episodes=1000, eval_episodes=100)
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
