import gymnasium as gym

ENVS = {
    'Pong': 'ALE/Pong-v5',
    'Breakout': 'ALE/Breakout-v5',
    'SpaceInvaders': 'ALE/SpaceInvaders-v5',
    'Asteroids': 'ALE/Asteroids-v5',
    'BattleZone': 'ALE/BattleZone-v5',
    'Boxing': 'ALE/Boxing-v5',
    'Centipede': 'ALE/Centipede-v5',
    'ChopperCommand': 'ALE/ChopperCommand-v5',
    'DoubleDunk': 'ALE/DoubleDunk-v5',
    'Enduro': 'ALE/Enduro-v5',
    'FishingDerby': 'ALE/FishingDerby-v5',
    'Freeway': 'ALE/Freeway-v5',
    'IceHockey': 'ALE/IceHockey-v5',
    'Jamesbond': 'ALE/Jamesbond-v5',
    'Kangaroo': 'ALE/Kangaroo-v5',
    'Krull': 'ALE/Krull-v5',
    'KungFuMaster': 'ALE/KungFuMaster-v5',
    'MsPacman': 'ALE/MsPacman-v5',
    'Qbert': 'ALE/Qbert-v5',
    'RoadRunner': 'ALE/RoadRunner-v5',
    'Seaquest': 'ALE/Seaquest-v5',
    'Tennis': 'ALE/Tennis-v5',
    'Tutankham': 'ALE/Tutankham-v5',
    'VideoPinball': 'ALE/VideoPinball-v5',
    'WizardOfWor': 'ALE/WizardOfWor-v5',
    'YarsRevenge': 'ALE/YarsRevenge-v5',
}

def get_env(env_name):
    return gym.make(ENVS[env_name], render_mode="rgb_array")
