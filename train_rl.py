from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from alexbotmini_env import AlexbotMiniEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardLoggerCallback(BaseCallback):
    def __init__(self, print_freq=1000, log_file='reward_log.txt', verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.rewards = []
        self.log_file = log_file
    def _on_step(self) -> bool:
        # 记录每个环境的奖励（数组）
        if 'reward' in self.locals:
            rewards = self.locals['reward']
        else:
            rewards = self.locals['rewards']
        # 展开并存储每个环境的 reward
        self.rewards.extend([float(r) for r in rewards])
        # 定期打印平均奖励
        if self.n_calls % self.print_freq == 0:
            avg_reward = sum(self.rewards[-self.print_freq:]) / self.print_freq
            print(f"Step {self.n_calls}: 最近{self.print_freq}步平均奖励: {avg_reward:.4f}")
            with open(self.log_file, 'a') as f:
                f.write(f"{self.n_calls},{avg_reward}\n")
        return True
def make_env():
    def _init():
        return AlexbotMiniEnv(render=False)
    return _init

if __name__ == '__main__':
    # 创建4个并行环境
    env = SubprocVecEnv([make_env() for _ in range(3)])

    model = PPO(
        'MlpPolicy',
        env,
        clip_range=0.2,
        learning_rate=0.0001,
        policy_kwargs={'net_arch': dict(pi=[64, 64], vf=[64, 64])},
        verbose=1,
        device='cpu'
    )
    # model = PPO.load('models/t_alexbotmini_ppo_600000_steps.zip', env=env, device='cpu', verbose=1)
    # 定义回调，每隔10万步保存一次模型
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/', name_prefix='alexbotmini_ppo')

    # 开始训练并传入回调
    reward_logger = RewardLoggerCallback(print_freq=10000)
    model.learn(total_timesteps=3000000, callback=[checkpoint_callback, reward_logger])
    model.save('alexbotmini_ppo_final')
