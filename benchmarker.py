import numpy as np
from .envs import get_env, ENVS
import requests
import json
import os

GITHUB_REPO_OWNER = "justbechit"
GITHUB_REPO_NAME = "rl_ladder"


def create_github_issue(title, body, labels=None):
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("GitHub token not found. Skipping issue creation.")
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


def benchmark(agent_class, pretrained=False, train_episodes=1000, eval_episodes=100):
    def wrapper(*args, **kwargs):
        agent = agent_class(*args, **kwargs)
        results = {}
        for env_name in ENVS:
            env = get_env(env_name)
            env_info = {
                'name': env_name,
                'action_space': env.action_space,
                'observation_space': env.observation_space
            }
            agent.set_env_info(env_info)
            
            if not pretrained:
                train_agent(agent, env, train_episodes)
            
            score = evaluate_agent(agent, env, eval_episodes)
            results[env_name] = score
        
        # github issue
        issue_title = f"Benchmark Results: {algo or agent_class.__name__}"
        issue_body = f"Name: {name}\n"
        issue_body += f"Algorithm: {algo or agent_class.__name__}\n"
        issue_body += "Benchmark results:\n\n"
        for env_name, score in results.items():
            issue_body += f"- {env_name}: {score:.2f}\n"
        issue_body += f"\nPretrained: {pretrained}\n"
        issue_body += f"Train episodes: {train_episodes}\n"
        issue_body += f"Eval episodes: {eval_episodes}\n"
        if comment:
            issue_body += f"\nComment: {comment}\n"
        create_github_issue(issue_title, issue_body, labels=["benchmark"])
        return results
    return wrapper

def train_agent(agent, env, episodes):
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

def evaluate_agent(agent, env, episodes=100, max_steps=10000):
    scores = []
    for _ in range(episodes):
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
    return np.mean(scores)

def run_benchmark(agent_class, pretrained=False, train_episodes=1000, eval_episodes=100, **agent_kwargs):
    agent = agent_class(**agent_kwargs)
    results = {}
    for env_name in ENVS:
        env = get_env(env_name)
        env_info = {
            'name': env_name,
            'action_space': env.action_space,
            'observation_space': env.observation_space
        }
        agent.set_env_info(env_info)
        
        if not pretrained:
            train_agent(agent, env, train_episodes)
        
        score = evaluate_agent(agent, env, eval_episodes)
        results[env_name] = score
     
    # 创建 GitHub issue
    issue_title = f"Benchmark Results: {algo or agent_class.__name__}"
    issue_body = f"Name: {name}\n"
    issue_body += f"Algorithm: {algo or agent_class.__name__}\n"
    issue_body += "Benchmark results:\n\n"
    for env_name, score in results.items():
        issue_body += f"- {env_name}: {score:.2f}\n"
    issue_body += f"\nPretrained: {pretrained}\n"
    issue_body += f"Train episodes: {train_episodes}\n"
    issue_body += f"Eval episodes: {eval_episodes}\n"
    if comment:
        issue_body += f"\nComment: {comment}\n"
    create_github_issue(issue_title, issue_body, labels=["benchmark"])
    return results


