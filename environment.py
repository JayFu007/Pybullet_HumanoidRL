import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import time  # 添加 time 模块

class URDFEnv(gym.Env):
    def __init__(self, urdf_path, render=True, robot_id=0, training_stage=1):
        super(URDFEnv, self).__init__()
        self.render = render
        self.urdf_path = urdf_path
        self.robot_id = robot_id  # 添加机器人编号
        self.target_distance = 10.0  # 修改目标距离为 10 米
        self.training_stage = training_stage  # 添加训练阶段

        # 使用 GUI 模式以启用可视化
        if self.render:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 隐藏默认 GUI
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()
        self.joint_limits = self._get_joint_limits()  # 获取关节限制

    def set_training_stage(self, stage):
        """
        设置当前训练阶段。
        Args:
            stage (int): 当前训练阶段（1-4）。
        """
        self.training_stage = stage

    def _get_joint_limits(self):
        # 获取每个关节的上下限
        joint_limits = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            lower_limit = joint_info[8]  # 下限
            upper_limit = joint_info[9]  # 上限
            joint_limits.append((lower_limit, upper_limit))
        return joint_limits

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)  # 确保重力设置正确

        # 创建地面
        plane_id = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, plane_id)

        # 加载机器人
        self.robot_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0.75])
        self.num_joints = p.getNumJoints(self.robot_id)

        # 初始化关节位置
        for joint_id in range(self.num_joints):
            p.resetJointState(self.robot_id, joint_id, targetValue=0.0)

        # 注释掉打印和可视化相关代码
        # base_position, _ = p.getBasePositionAndOrientation(self.robot_id)
        # print(f"Base initial height: {base_position[2]}")
        # self._add_world_axes()

        # 定义动作空间和观察空间
        self.forward_joints = [0, 3, 6, 9, 5, 11]  # 仅允许活动的关节索引
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(len(self.forward_joints),), dtype=np.float32)  # 缩小动作范围
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints * 2,), dtype=np.float32)

        # 设置目标位置为 (0, 10, 0)
        self.target_position = [0, self.target_distance, 0]

        # 延迟 1 秒
        if self.render:
            time.sleep(1)  # 延迟以确保可视化窗口加载完成

        return self._get_observation()

    # 注释掉世界坐标系可视化
    # def _add_world_axes(self):
    #     axis_length = 1.0
    #     p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], 2, physicsClientId=self.client)
    #     p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], 2, physicsClientId=self.client)
    #     p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], 2, physicsClientId=self.client)

    def render_robot_id(self):
        # 移除对机器人名称的标注
        pass

    def step(self, action):
        scaled_action = np.clip(action, -0.5, 0.5)
        max_force = 150
        joint_torques = []  # 记录关节力矩
        for idx, joint_id in enumerate(self.forward_joints):
            current_position = p.getJointState(self.robot_id, joint_id)[0]
            target_position = current_position + scaled_action[idx]
            lower_limit, upper_limit = self.joint_limits[joint_id]
            target_position = np.clip(target_position, lower_limit, upper_limit)
            p.setJointMotorControl2(
                self.robot_id, joint_id, p.POSITION_CONTROL,
                targetPosition=target_position, force=max_force,
                positionGain=0.1, velocityGain=0.1
            )
            joint_torques.append(p.getJointState(self.robot_id, joint_id)[3])  # 获取力矩
        p.stepSimulation()

        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_angles = [state[0] for state in joint_states]  # 获取关节角度
        joint_torques = [state[3] for state in joint_states]  # 获取关节力矩

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()

        info = {
            "joint_angles": joint_angles,  # 包含所有关节的角度
            "joint_torques": joint_torques,  # 包含所有关节的力矩
            "visualize_reward": reward  # 将奖励值添加到 info
        }

        if self.render:
            time.sleep(0.02)

        return obs, reward, done, info

    def _get_observation(self):
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return np.array(joint_positions + joint_velocities, dtype=np.float32)

    def _compute_reward(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)

        # 阶段 1：学习稳定站立
        if self.training_stage == 1:
            roll, pitch, _ = p.getEulerFromQuaternion(base_orientation)
            orientation_penalty = -abs(roll) - abs(pitch)
            height = base_position[2]
            height_reward = -abs(height - 0.75)  # 奖励基座高度接近初始值
            return 0.5 * orientation_penalty + height_reward

        # 阶段 2：尝试移动
        elif self.training_stage == 2:
            distance_to_target = np.linalg.norm(np.array(base_position[:2]) - np.array(self.target_position[:2]))
            distance_reward = max(0, self.target_distance - distance_to_target)
            roll, pitch, _ = p.getEulerFromQuaternion(base_orientation)
            orientation_penalty = -abs(roll) - abs(pitch)
            return distance_reward + 0.5 * orientation_penalty

        # 阶段 3：优化步态
        elif self.training_stage == 3:
            distance_to_target = np.linalg.norm(np.array(base_position[:2]) - np.array(self.target_position[:2]))
            distance_reward = max(0, self.target_distance - distance_to_target)
            joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
            joint_velocities = [state[1] for state in joint_states]
            smoothness_penalty = -np.sum(np.square(joint_velocities)) * 0.01
            return distance_reward + smoothness_penalty

        # 阶段 4：高效抵达目标
        elif self.training_stage == 4:
            distance_to_target = np.linalg.norm(np.array(base_position[:2]) - np.array(self.target_position[:2]))
            distance_reward = max(0, self.target_distance - distance_to_target)
            time_penalty = -0.1  # 每步惩罚时间
            return distance_reward + time_penalty

        return 0

    def _check_done(self):
        base_position, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance_to_target = np.linalg.norm(np.array(base_position[:2]) - np.array(self.target_position[:2]))
        if distance_to_target < 0.1:  # 如果机器人接近目标位置，任务完成
            return True
        if base_position[2] < 0.3 or base_position[2] > 1.0:  # 基座高度异常，认为机器人倒地或飞起
            return True
        return False

