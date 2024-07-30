import numpy as np
from tqdm import tqdm
import requests
import json
import time
from functools import wraps
from just_bench_it.envs import get_env, ENVS

GITHUB_REPO_OWNER = "justbechit"
GITHUB_REPO_NAME = "rl_ladder.github.io"
GITHUB_CLIENT_ID = "Ov23li6k9dJ9Ws9bsWy0"
GITHUB_CLIENT_SECRET = "5c8ab1c55c159ecb683d7fbbfe1efe657dc1d536"

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
            
            def bench(self):
                results = {}

                for env_name in tqdm(ENVS, desc="Environments"):
                    env = get_env(env_name)
                    env_info = {
                        'name': env_name,
                        'action_space': env.action_space,
                        'observation_space': env.observation_space,
                        'env': env,
                        'train_episodes': train_episodes,
                        'eval_episodes': eval_episodes
                    }
                    self.set_env_info(env_info)

                    if not pretrained:
                        train_agent(self, env, train_episodes)
                    
                    score = evaluate_agent(self, env, eval_episodes)
                    results[env_name] = score
                    print(f"Environment: {env_name}, Score: {score:.2f}")

                # Create GitHub issue
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

def train_agent(agent, env, episodes, max_steps=100):
    total_reward = 0
    pbar = tqdm(range(episodes), desc="Training")
    for episode in pbar:
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
        pbar.set_description(f"Training (Avg Reward: {avg_reward:.2f})")
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}")

def evaluate_agent(agent, env, episodes=100, max_steps=10000):
    scores = []
    pbar = tqdm(range(episodes), desc="Evaluating")
    for episode in pbar:
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
        pbar.set_description(f"Evaluating (Avg Score: {np.mean(scores):.2f})")

        if (episode + 1) % 10 == 0:
            print(f"Evaluation Episode {episode + 1}/{episodes}, Average Score: {np.mean(scores):.2f}")

    return np.mean(scores)
