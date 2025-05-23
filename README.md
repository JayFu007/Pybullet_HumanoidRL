# 基于Pybullet打造的机器人强化学习环境：Pybullet_HumanoidRL
本项目是一个基于Pybullet的强化学习环境，主要用于训练和测试人形机器人（Humanoid Robot）的强化学习算法。该项目的目标是为研究人员和开发者提供一个易于使用和扩展的框架，以便在Pybullet中进行人形机器人相关的强化学习实验。



# 项目结构  
  
├── agent.py             # PPO算法 + Actor-Critic模型  
├── environment.py         # PyBullet环境 + 奖励/终止逻辑  
├── train.py              # 训练入口脚本  
├── utils.py             # 模型评估工具  
└── alexbotmini.urdf     # 机器人URDF模型  

# 更新日志
****
本项目于2025年5月18日更新第一个版本，目前仍在维护；
后续版本将会添加更多功能和改进。  
我欢迎任何反馈和建议，以帮助我改进这个项目。  
目前已经完成了对以下功能的实现：  
机器人的可视化仿真渲染、机器人的训练环境、强化学习算法基础。