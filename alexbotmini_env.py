import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np

class AlexbotMiniEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)
        self.render = render
        self.training_steps = 0  # 初始化训练步数
        if self.render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = None
        self.reset()

    # def _adjust_weights(self):
    #     # 根据训练步数动态调整权重
    #     max_steps = 3000000  # 假设总训练步数
    #     progress = min(self.training_steps / max_steps, 1.0)  # 归一化进度
    #     forward_weight = 0.5 + 0.3 * progress  # 前进奖励从0.5增加到1.0
    #     balance_weight = 1.0 - 0.3 * progress  # 平衡奖励从1.0减少到0.5
    #     return forward_weight, balance_weight

    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭PyBullet GUI面板
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.setGravity(0, 0, -9.8 + np.random.uniform(-0.1, 0.1))
        p.loadURDF('plane.urdf')
        self.robot = p.loadURDF('alexbotmini.urdf', [0, 0, 0.735],p.getQuaternionFromEuler([0,0,-np.pi/2]))
        for i in range(12):
            p.resetJointState(self.robot, i, targetValue=np.random.uniform(-0.1, 0.1))
        # 获取末端 link 的 id
        def get_link_index_by_name(robot, link_name):
            for i in range(p.getNumJoints(robot)):
                info = p.getJointInfo(robot, i)
                if info[12].decode('utf-8') == link_name:
                    return i
            raise ValueError(f"Link name {link_name} not found.")
        self.left_foot_id = get_link_index_by_name(self.robot, 'leftlink6')
        self.right_foot_id = get_link_index_by_name(self.robot, 'rightlink6')
        self.prev_base_pos = p.getBasePositionAndOrientation(self.robot)[0]
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.training_steps += 1
        joint_names = ['leftjoint1','leftjoint4', 'leftjoint6',
                       'rightjoint1','rightjoint4', 'rightjoint6']
        for i, name in enumerate(joint_names):
            joint_id = [p.getJointInfo(self.robot, j)[1].decode() for j in range(p.getNumJoints(self.robot))].index(name)
            p.setJointMotorControl2(self.robot, joint_id, p.POSITION_CONTROL, targetPosition=action[i])
        p.stepSimulation()
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        if done:
            reward -= 10.0
        info = {}
        return obs, reward, done, info
    # def _get_obs(self):
    #     # 观测：6个关节角度+6个速度
    #     joint_states = [p.getJointState(self.robot, i) for i in range(6)]
    #     pos = [s[0] for s in joint_states]
    #     vel = [s[1] for s in joint_states]
    #     return np.array(pos + vel, dtype=np.float32)

    def _get_obs(self):
        # 关节状态（12维）
        joint_states = [p.getJointState(self.robot, i) for i in range(12)]  # 原为 range(6)
        pos = [s[0] for s in joint_states]
        vel = [s[1] for s in joint_states]

        # 基座状态（6维：位置xyz + 欧拉角roll/pitch）
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(base_ori)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot)

        # 合并观测（总维度：12+12+6+3+3=36）
        obs = np.concatenate([
            pos, vel,
            base_pos,
            euler,
            base_lin_vel,
            base_ang_vel
        ], dtype=np.float32)
        # print("obs shape:", obs.shape)  # 输出观测空间的形状
        return obs

    # def _compute_reward(self):
    #     base_pos = p.getBasePositionAndOrientation(self.robot)[0]
    #     base_orientation = p.getBasePositionAndOrientation(self.robot)[1]
    #     euler = p.getEulerFromQuaternion(base_orientation)
    #
    #     # 前进奖励
    #     forward_velocity = base_pos[0] - self.prev_base_pos[0]
    #     forward_reward = forward_velocity - abs(forward_velocity - 0.1)  # 目标速度为0.1
    #
    #     # 平衡性奖励
    #     balance_reward = -abs(base_pos[1]) - abs(euler[0]) - abs(euler[1])
    #
    #     # 能耗惩罚
    #     joint_states = [p.getJointState(self.robot, i) for i in range(6)]
    #     energy_penalty = sum((s[3] ** 2 + s[1] ** 2) for s in joint_states)
    #
    #     # 步态周期性奖励
    #     left_foot_pos = p.getLinkState(self.robot, self.left_foot_id)[0]
    #     right_foot_pos = p.getLinkState(self.robot, self.right_foot_id)[0]
    #     gait_reward = -abs(left_foot_pos[2] - right_foot_pos[2]) - abs(left_foot_pos[0] - right_foot_pos[0])
    #
    #     # 稳定性奖励
    #     stability_reward = -abs(base_pos[2] - 0.74)  # 目标高度为0.74
    #
    #     # 动态调整权重
    #     forward_weight, balance_weight = self._adjust_weights()
    #     jump_penalty = 0.0
    #     if base_pos[2] > 0.8:  # 0.74为正常高度，0.78为跳跃阈值
    #         jump_penalty = -10.0 * (base_pos[2] - 0.8)  # 惩罚幅度可调整
    #
    #     # 奖励缩放
    #     reward = 0.1 * (
    #         forward_weight * forward_reward +
    #         balance_weight * balance_reward -
    #         0.0005 * energy_penalty +   # 能耗惩罚缩小
    #         0.4 * gait_reward +         # 步态奖励提升
    #         0.5 * stability_reward +    # 稳定性奖励提升
    #         jump_penalty
    #     )
    #     self.prev_base_pos = base_pos
    #     return reward

    def _compute_reward(self):
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        base_lin_vel = p.getBaseVelocity(self.robot)[0]  # 新增基座线速度
        euler = p.getEulerFromQuaternion(base_ori)

        # 1. 前进奖励：鼓励正向速度，并设定合理目标速度
        forward_velocity = base_lin_vel[0]  # 使用瞬时速度代替位置差
        forward_reward = np.clip(forward_velocity, -0.5, 0.5)  # 限制速度范围避免异常值

        # 2. 平衡奖励：惩罚倾斜角度和横向偏移
        balance_penalty = (abs(euler[0]) + abs(euler[1]) ** 2) * 0.5  # 俯仰角惩罚加重

        # 3. 能耗惩罚：降低惩罚系数
        joint_states = [p.getJointState(self.robot, i) for i in range(6)]
        energy_penalty = sum(abs(s[3]) for s in joint_states) * 0.01  # 使用扭矩绝对值

        # 4. 简化其他奖励项
        stability_penalty = abs(base_pos[2] - 0.74) * 0.5  # 高度稳定性

        # 动态权重调整（更平缓）
        forward_weight, balance_weight = self._adjust_weights()

        # 总奖励公式
        reward = (
                forward_weight * forward_reward
                - balance_weight * balance_penalty
                - energy_penalty
                - stability_penalty
        )

        # 跳跃惩罚（仅在严重异常时触发）
        if base_pos[2] > 0.8:
            reward -= 10.0
        return reward

    def _adjust_weights(self):
        # 调整权重变化曲线为线性平缓
        progress = min(self.training_steps / 3000000, 1.0)
        forward_weight = 0.3 + 0.7 * progress  # 0.3 -> 1.0
        balance_weight = 1.0 - 0.2 * progress  # 1.0 -> 0.8
        return forward_weight, balance_weight

    def _is_done(self):
        # 终止条件：机器人摔倒
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        if base_pos[2] < 0.4 or abs(base_pos[1]) > 0.5:  # 添加横向偏移终止条件
            return True
        return False

    def render(self, mode='human'):
        pass