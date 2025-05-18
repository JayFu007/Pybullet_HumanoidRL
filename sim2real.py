import torch
import numpy as np
from environment import URDFEnv

# 加载通过 torch.jit 保存的模型
model = torch.jit.load("policy_6000.pt")
model.eval()

# 创建 URDF 环境
env = URDFEnv(urdf_path="alexbotmini.urdf", render=True)
state = env.reset()

done = False
while not done:
    # 将状态转换为张量并确保形状正确
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加 batch 维度
    state_tensor = state_tensor[:, :24]  # 确保输入维度为 24（根据模型定义）

    with torch.no_grad():
        action_mean, _, _ = model(state_tensor)
    action = action_mean.squeeze(0).numpy()

    # 与环境交互
    next_state, reward, done, info = env.step(action)
    state = next_state

    # 打印关节角度和力矩信息
    print(f"Joint Angles: {info['joint_angles']}")
    print(f"Joint Torques: {info['joint_torques']}")
