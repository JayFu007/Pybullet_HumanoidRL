from environment import URDFEnv
from agent import PPO
import pybullet as p
import numpy as np
import torch
def evaluate_trained_model(urdf_path, model_path, num_episodes=5):
    """
    加载训练好的模型并在环境中运行测试以查看训练效果。
    """
    env = URDFEnv(urdf_path=urdf_path, render=True, robot_id=0)  # 启用可视化
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 初始化 PPO 智能体并加载模型
    agent = PPO(state_dim, action_dim)
    agent.policy.load_state_dict(torch.load(model_path))
    agent.policy.eval()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    input("Press Enter to close the visualization...")  # 保持窗口打开

def visualize_joint_info(urdf_path, joint_indices=None):
    """
    可视化URDF文件中关节的名称及其活动的三维坐标轴。

    Args:
        urdf_path (str): URDF文件路径。
        joint_indices (list, optional): 要显示的关节索引列表。如果为None，则显示所有关节。
    """
    # 加载机器人模型
    physics_client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 禁用默认GUI
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # 禁用阴影
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])

    robot_id = p.loadURDF(urdf_path)

    num_joints = p.getNumJoints(robot_id)
    print(f"机器人共有 {num_joints} 个关节。")

    # 遍历关节并显示信息
    for i in range(num_joints):
        if joint_indices is not None and i not in joint_indices:
            continue

        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")
        joint_axis = joint_info[13]  # 获取关节活动坐标轴
        joint_position = p.getLinkState(robot_id, i)[0]  # 获取关节位置

        print(f"关节索引: {i}, 名称: {joint_name}, 活动坐标轴: {joint_axis}")

        # 可视化关节的三维活动坐标轴
        axis_length = 0.2
        x_axis = [axis_length, 0, 0]
        y_axis = [0, axis_length, 0]
        z_axis = [0, 0, axis_length]

        # 绘制X轴（红色）
        p.addUserDebugLine(joint_position, [joint_position[j] + x_axis[j] for j in range(3)], [1, 0, 0], 2)
        # 绘制Y轴（绿色）
        p.addUserDebugLine(joint_position, [joint_position[j] + y_axis[j] for j in range(3)], [0, 1, 0], 2)
        # 绘制Z轴（蓝色）
        p.addUserDebugLine(joint_position, [joint_position[j] + z_axis[j] for j in range(3)], [0, 0, 1], 2)

        # 可视化关节名称
        p.addUserDebugText(joint_name, joint_position, textColorRGB=[1, 0, 0], textSize=1.5)

    input("按回车键关闭可视化...")
    p.disconnect()
if __name__ == "__main__":
    visualize_joint_info(urdf_path="alexbotmini.urdf", joint_indices=[0, 3, 6, 9, 5, 11])