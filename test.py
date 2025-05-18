import visualize
from agent import PPO
from visualize import *
import numpy as np
import torch
import time
import io
from utils import *
def train_single_robot(urdf_path, num_episodes, save_every, model_path=None):
    env = URDFEnv(urdf_path=urdf_path, render=True, robot_id=0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)

    if model_path:
        agent.policy.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")

    current_stage = 1
    stage_durations = [10000, 10000, 10000, 10000]
    stage_thresholds = np.cumsum(stage_durations)

    rewards_history = []
    for episode in range(num_episodes):
        if episode >= stage_thresholds[current_stage - 1]:
            current_stage += 1
            env.set_training_stage(current_stage)
            print(f"Switching to training stage {current_stage}")

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            info["visualize_reward"] = reward
            agent.store_transition((state, action, reward, done, info))
            state = next_state
            total_reward += reward

        agent.train()
        rewards_history.append(total_reward)

        print(f"Episode {episode + 1}/{num_episodes}, Stage {current_stage}, Total Reward: {total_reward}")

        if (episode + 1) % save_every == 0:
            torch.save(agent.policy.state_dict(), f"ppo_model_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")

    torch.save(agent.policy.state_dict(), "ppo_final_model.pth")
    print("Training completed. Final model saved.")

if __name__ == "__main__":

    train_single_robot(urdf_path="alexbotmini.urdf", num_episodes=40000, save_every=10000, model_path=None)
    # evaluate_trained_model(urdf_path="alexbotmini.urdf",model_path=r"H:\temp\humanoid\Humanoid-Robot-Reinforcement-Learning-PPO\bt_data\ppo_sfinal_model.pth",num_episodes=5)