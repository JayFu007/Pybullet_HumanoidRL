import pybullet as p
import pybullet_data
import time
import numpy as np

# 连接到PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF('plane.urdf')
robot = p.loadURDF('alexbotmini.urdf', [0, 0, 0.735], p.getQuaternionFromEuler([0, 0, -np.pi/2]))

# 获取关节信息
joint_name_to_id = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}
# joint_names = [
#     'leftjoint1', 'leftjoint2', 'leftjoint3', 'leftjoint4', 'leftjoint5', 'leftjoint6',
#     'rightjoint1', 'rightjoint2', 'rightjoint3', 'rightjoint4', 'rightjoint5', 'rightjoint6'
# ]
joint_names = ['leftjoint1', 'leftjoint4', 'leftjoint6',
              'rightjoint1', 'rightjoint4', 'rightjoint6']
# 为每个关节添加滑块
sliders = []
for name in joint_names:
    sliders.append(p.addUserDebugParameter(name, -1, 1, 0))

# PID控制器参数
kp = 1.0  # 比例增益
kd = 0.1  # 微分增益
ki = 0.0  # 积分增益
prev_error = [0.0, 0.0]  # 俯仰角和横滚角的前一误差
integral_error = [0.0, 0.0]

while True:
    # 获取基座状态
    base_pos, base_ori = p.getBasePositionAndOrientation(robot)
    euler = p.getEulerFromQuaternion(base_ori)  # 转换为欧拉角
    pitch_error = -euler[1]  # 俯仰角误差（目标为0）
    roll_error = -euler[0]  # 横滚角误差（目标为0）

    # PID控制计算
    pitch_control = kp * pitch_error + kd * (pitch_error - prev_error[0]) + ki * integral_error[0]
    roll_control = kp * roll_error + kd * (roll_error - prev_error[1]) + ki * integral_error[1]

    # 更新误差
    prev_error = [pitch_error, roll_error]
    integral_error[0] += pitch_error
    integral_error[1] += roll_error

    # 读取滑块值并叠加平衡控制
    for i, name in enumerate(joint_names):
        slider_val = p.readUserDebugParameter(sliders[i])
        joint_id = joint_name_to_id[name]
        if 'left' in name:
            control_val = slider_val + pitch_control + roll_control
        elif 'right' in name:
            control_val = slider_val + pitch_control - roll_control
        else:
            control_val = slider_val
        p.setJointMotorControl2(robot, joint_id, p.POSITION_CONTROL, targetPosition=control_val)

    p.stepSimulation()
    time.sleep(1./240.)