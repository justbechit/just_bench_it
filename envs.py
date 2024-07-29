import gymnasium as gym

ENVS = {
    'Pong': 'PongNoFrameskip-v4',
    'Breakout': 'BreakoutNoFrameskip-v4',
    'SpaceInvaders': 'SpaceInvadersNoFrameskip-v4',
    # 可以添加更多 Atari 游戏
}

def get_env(env_name):
    return gym.make(ENVS[env_name])
