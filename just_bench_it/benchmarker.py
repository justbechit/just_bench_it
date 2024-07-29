import numpy as np
from just_bench_it.envs import get_env, ENVS
import requests
import json

GITHUB_REPO_OWNER = "justbechit"
GITHUB_REPO_NAME = "rl_ladder"
GITHUB_CLIENT_ID = "Ov23li6k9dJ9Ws9bsWy0"
GITHUB_CLIENT_SECRET = "5c8ab1c55c159ecb683d7fbbfe1efe657dc1d536"

def get_github_token():
    url = "https://github.com/login/oauth/access_token"
    headers = {"Accept": "application/json"}
    data = {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "grant_type": "client_credentials"
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"Failed to obtain GitHub token: {response.content}")
        return None

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


def benchmark(pretrained=False, train_episodes=1000, eval_episodes=100):
    def decorator(agent_class):
        def wrapper(*args, **kwargs):
            return run_benchmark(agent_class, pretrained, train_episodes, eval_episodes, *args, **kwargs)
        return wrapper
    return decorator

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
    issue_title = f"Benchmark Results: {agent_class.__name__}"
    issue_body = f"Algorithm: {agent_class.__name__}\n"
    issue_body += "Benchmark results:\n\n"
    for env_name, score in results.items():
        issue_body += f"- {env_name}: {score:.2f}\n"
    issue_body += f"\nPretrained: {pretrained}\n"
    issue_body += f"Train episodes: {train_episodes}\n"
    issue_body += f"Eval episodes: {eval_episodes}\n"
    if agent_kwargs.get('comment'):
        issue_body += f"\nComment: {agent_kwargs['comment']}\n"
    create_github_issue(issue_title, issue_body, labels=["benchmark"])
    return results

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