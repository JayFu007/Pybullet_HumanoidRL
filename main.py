import pybullet as p
import pybullet_data
import time
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF('plane.urdf')
p.setGravity(0, 0, -9.8)
robot = p.loadURDF('alexbotmini.urdf', [0, 0, 0.74])
num_joints = p.getNumJoints(robot)

# 12个关节：右腿0-5，左腿6-11
amplitudes = [0.2] * 12
frequencies = [0.2] * 12
phases = [0, math.pi, 0, math.pi, 0, math.pi,  # 右腿
          math.pi, 0, math.pi, 0, math.pi, 0]  # 左腿

# 髋关节前倾偏置（假设第0和6号关节为髋关节）
bias = [0.2] + [0]*5 + [0.2] + [0]*5

def get_robot_tilt():
    base_orientation = p.getBasePositionAndOrientation(robot)[1]
    euler = p.getEulerFromQuaternion(base_orientation)
    return euler[1]

for t in range(4000):
    time_in_seconds = t / 240.0
    tilt = get_robot_tilt()
    for i in range(num_joints):
        adjusted_amplitude = amplitudes[i] * (1 - 0.5 * abs(tilt))
        target = bias[i] + adjusted_amplitude * math.sin(2 * math.pi * frequencies[i] * time_in_seconds + phases[i])
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=target, force=20)
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()