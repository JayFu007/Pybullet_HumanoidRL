import time
from stable_baselines3 import PPO
from alexbotmini_env import AlexbotMiniEnv

env = AlexbotMiniEnv(render=True)
model = PPO.load('models/alexbotmini_ppo_300000_steps.zip')

total_reward = 0
total_steps = 0

obs = env.reset()
for _ in range(100000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    total_steps += 1
    time.sleep(1./240.)
    if done:
        obs = env.reset()
        print(f"平均奖励: {total_reward / total_steps}")