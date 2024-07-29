import gymnasium as gym

ENVS = {
    'Pong': 'ALE/Pong-v5',
    'Breakout': 'ALE/Breakout-v5',
    'SpaceInvaders': 'ALE/SpaceInvaders-v5',
    # 可以添加更多 Atari 游戏
}

def get_env(env_name):
    return gym.make(ENVS[env_name], render_mode="rgb_array")
