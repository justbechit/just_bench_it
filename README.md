
# Just Bench It: RL Algorithm Benchmarking Tool

这个项目提供了一个简单的工具，用于对强化学习（RL）算法在Atari游戏上进行基准测试。
WEBSITE: https://justbechit.github.io/rl_ladder/

## 安装
## PYPI

1. 安装：
   ```
   pip install just-bench-it
   ```

### Build from source

1. 克隆这个仓库：
   ```
   git clone https://github.com/your_username/just_bench_it.git
   cd just_bench_it
   ```

2. 安装依赖：
   ```
   pip install -e .
   ```

## 使用方法

1. 创建你的RL agent类，并使用`@benchmark`装饰器。

2. 在你的agent类中实现以下方法：
   - `set_env_info(self, env_info)`: 设置环境信息
   - `act(self, state)`: 根据当前状态选择动作
   - `update(self, state, action, reward, next_state, done)`: 更新agent的内部状态或模型

3. 运行你的脚本来执行基准测试。

## 示例

这里有一个DQN agent的示例实现：

```python
from just_bench_it import benchmark

@benchmark(pretrained=False, train_episodes=1000, eval_episodes=100)
class DQNAgent:
    def __init__(self):
        # 初始化你的DQN agent
        pass

    def set_env_info(self, env_info):
        # 设置环境信息： bench_it 会提供当前动作空间和观察空间
        #         input_shape = env_info['observation_space'].shape
        #         output_dim = env_info['action_space'].n
        #  不同的环境其输入可能不同，确保您的算法能够应对不同环境
        pass

    def act(self, state):
        # 根据状态选择动作
        pass

    def update(self, state, action, reward, next_state, done):
        # 更新agent
        pass

if __name__ == "__main__":
    agent = DQNAgent()
    results = agent.bench()
    print(results)
```

## 自定义

你可以通过修改`@benchmark`装饰器的参数来自定义基准测试：

- `pretrained`: 是否使用预训练模型（默认为False）
- `train_episodes`: 训练的回合数（默认为1000）
- `eval_episodes`: 评估的回合数（默认为100）

## 结果

基准测试的结果会自动发布为GitHub issue，包含每个环境的平均得分和其他相关信息。

## 贡献

欢迎提交问题报告和拉取请求。对于重大更改，请先开issue讨论您想要更改的内容。

## 许可证

[MIT](https://choosealicense.com/licenses/mit/)
