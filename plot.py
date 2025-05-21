# 读取日志文件并绘制奖励变化图
import matplotlib.pyplot as plt

def plot_rewards(log_file='s_reward_log.txt'):
    steps = []
    rewards = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                step, reward = map(float, line.split(','))
                steps.append(step)
                rewards.append(reward)
            except ValueError:
                continue  # 跳过格式错误的行
    plt.plot(steps, rewards)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Reward Function Over Time')
    plt.show()
# 调用绘图函数
plot_rewards('s_reward_log.txt')