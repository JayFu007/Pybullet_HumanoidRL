import pybullet as p
import pybullet_data
import time
import numpy as np
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭PyBullet GUI面板
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_COORDINATE_FRAME, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8 + np.random.uniform(-0.1, 0.1))
p.loadURDF('plane.urdf')
robot = p.loadURDF('alexbotmini.urdf', [0, 0, 0.735],p.getQuaternionFromEuler([0, 0, -np.pi/2]))

joint_name_to_id = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}

# 创建滑块
left_slider = p.addUserDebugParameter('leftjoint1', -1, 1, 0)
right_slider = p.addUserDebugParameter('rightjoint1', -1, 1, 0)

while True:
    left_val = p.readUserDebugParameter(left_slider)
    right_val = p.readUserDebugParameter(right_slider)
    p.setJointMotorControl2(robot, joint_name_to_id['leftjoint1'], p.POSITION_CONTROL, targetPosition=left_val)
    p.setJointMotorControl2(robot, joint_name_to_id['rightjoint1'], p.POSITION_CONTROL, targetPosition=right_val)
    p.stepSimulation()
    time.sleep(1./240.)