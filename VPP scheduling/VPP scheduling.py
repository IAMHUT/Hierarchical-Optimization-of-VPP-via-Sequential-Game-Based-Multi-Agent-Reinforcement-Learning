import torch
import torch.nn as nn
import torch.optim as optim  # 添加这行
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import copy
from scipy import stats
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 博弈模型参数 ====================
c_h, c_m, c_l = 0.3, 0.2, 0.1  # 高、中、低价 (元/kWh)
alpha = 0.99  # 单位成本惩罚系数
beta = 180.0  # 供电侧机组启停成本 (修改为单位成本)
N = 100  # 负载数量
T = 1.0  # 总时间，归一化


# ==================== 博弈环境（供电侧与负载侧） ====================
class PowerGridEnv:
    def __init__(self, time_slots=12):
        self.c = [c_h, c_m, c_l]  # 价格
        self.alpha = alpha  # 单位成本惩罚系数
        self.beta = beta  # 机组启停成本
        self.N = N  # 负载数量
        self.T = T  # 总时间
        self.time_slots = time_slots

        # 十二个时间段的基础电价概率
        self.time_based_price_probs = np.array([0.12, 0.08, 0.15, 0.25, 0.62, 0.88,
                                                0.95, 0.92, 0.88, 0.73, 0.45, 0.18])

    def generate_price_strategy(self, p_high, p_mid, p_low):
        """生成十二个时间段的供电侧策略"""
        # p_high, p_mid, p_low 是每个时间段选择高、中、低电价的概率
        # 转换为十二个时间段的策略
        strategies = []
        for t in range(self.time_slots):
            # 根据时间特性调整策略
            time_factor = self.time_based_price_probs[t]
            # 高电价概率与时间因子相关
            p_h_adjusted = p_high[t] * time_factor
            p_m_adjusted = p_mid[t] * (1 - time_factor) * 0.7
            p_l_adjusted = p_low[t] * (1 - time_factor) * 0.3

            # 归一化
            total = p_h_adjusted + p_m_adjusted + p_l_adjusted + 1e-10
            strategies.append([p_h_adjusted / total, p_m_adjusted / total, p_l_adjusted / total])

        return np.array(strategies)

    def generate_load_strategy(self, q_high, q_mid, q_low):
        """生成十二个时间段的负载侧策略"""
        strategies = []
        for t in range(self.time_slots):
            # 负载策略也需要考虑时间特性
            time_factor = self.time_based_price_probs[t]
            # 在高电价时段，负载倾向于减少用电
            q_h_adjusted = q_high[t] * (1 - time_factor * 0.8)
            q_m_adjusted = q_mid[t] * 0.5
            q_l_adjusted = q_low[t] * (1 + time_factor * 0.5)

            total = q_h_adjusted + q_m_adjusted + q_l_adjusted + 1e-10
            strategies.append([q_h_adjusted / total, q_m_adjusted / total, q_l_adjusted / total])

        return np.array(strategies)

    def step(self, p_strategies, q_strategies):
        """执行博弈一步"""
        # p_strategies: (12, 3) 供电侧策略
        # q_strategies: (12, 3) 负载侧策略

        total_U_s = 0  # 供电侧总收益
        total_U_v = 0  # 负载侧总收益

        # 计算每个时间段的博弈结果
        for t in range(self.time_slots):
            p = p_strategies[t]
            q = q_strategies[t]

            # 模拟负载选择
            k = np.random.multinomial(self.N // self.time_slots, q)

            # 供电侧收益
            U_s = sum(p[i] * (self.c[i] - self.beta / 1000) * k[i] for i in range(3))

            # 负载侧成本（考虑单位成本惩罚系数）
            x = [np.clip(0.9 - (self.N // self.time_slots) * p[i] / max(k[i], 1), 0, 0.9) ** 2 for i in range(3)]
            U_v = -sum(q[i] * (self.c[i] + self.alpha * x[i]) for i in range(3))

            total_U_s += U_s
            total_U_v += U_v

        # 返回状态：平均策略
        avg_p = p_strategies.mean(axis=0)
        avg_q = q_strategies.mean(axis=0)

        state = np.concatenate([avg_p, avg_q, self.c])
        return state, total_U_s, total_U_v


# ==================== 博弈智能体（MADDPG） ====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)

        self.gamma = 0.99
        self.tau = 0.01

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).numpy()
        return action

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)

        # 更新 Critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train_game_theory_model():
    """训练博弈模型"""
    env = PowerGridEnv(time_slots=12)
    state_dim = 9  # [p_high_avg, p_mid_avg, p_low_avg, q_high_avg, q_mid_avg, q_low_avg, c_h, c_m, c_l]
    action_dim = 3  # 每个智能体的动作维度

    # 供电侧和负载侧智能体
    supply_agent = MADDPGAgent(state_dim, action_dim)
    load_agent = MADDPGAgent(state_dim, action_dim)

    episodes = 100
    max_steps = 50
    batch_size = 32
    memory = []

    print("开始训练博弈模型...")

    for episode in range(episodes):
        # 初始状态
        state = np.concatenate([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [c_h, c_m, c_l]])
        total_supply_reward = 0
        total_load_reward = 0

        for step in range(max_steps):
            # 选择动作（基础策略）
            supply_base_action = supply_agent.select_action(state)
            load_base_action = load_agent.select_action(state)

            # 扩展为十二个时间段的策略
            # 为简化，我们假设每个时间段的策略相同，但加入时间波动
            time_factors = np.linspace(0.8, 1.2, 12)
            p_high = supply_base_action[0] * time_factors
            p_mid = supply_base_action[1] * np.ones(12)
            p_low = supply_base_action[2] * (2 - time_factors)

            q_high = load_base_action[0] * (2 - time_factors)
            q_mid = load_base_action[1] * np.ones(12)
            q_low = load_base_action[2] * time_factors

            # 生成完整的十二时间段策略
            p_strategies = env.generate_price_strategy(p_high, p_mid, p_low)
            q_strategies = env.generate_load_strategy(q_high, q_mid, q_low)

            # 环境交互
            next_state, supply_reward, load_reward = env.step(p_strategies, q_strategies)

            # 存储经验
            memory.append((state, np.concatenate([supply_base_action, load_base_action]),
                           [supply_reward, load_reward], next_state, 0))

            state = next_state
            total_supply_reward += supply_reward
            total_load_reward += load_reward

            # 经验回放更新
            if len(memory) >= batch_size:
                batch = np.random.choice(len(memory), batch_size, replace=False)
                states = np.array([memory[i][0] for i in batch])
                actions = np.array([memory[i][1] for i in batch])
                rewards = np.array([memory[i][2] for i in batch])
                next_states = np.array([memory[i][3] for i in batch])
                dones = np.array([memory[i][4] for i in batch])

                # 更新负载智能体
                load_agent.update(states, actions[:, 3:], rewards[:, 1], next_states, dones)
                # 更新供电智能体
                supply_agent.update(states, actions[:, :3], rewards[:, 0], next_states, dones)

        if episode % 20 == 0:
            print(
                f"Episode {episode + 1}, Supply Reward: {total_supply_reward:.2f}, Load Reward: {total_load_reward:.2f}")

    # 获取最终均衡策略
    final_state = torch.FloatTensor(state).unsqueeze(0)
    supply_final_action = supply_agent.actor(final_state).squeeze(0).detach().numpy()
    load_final_action = load_agent.actor(final_state).squeeze(0).detach().numpy()

    print(f"均衡策略 - 供电侧: {supply_final_action}, 负载侧: {load_final_action}")

    # 生成十二个时间段的高电价概率
    time_factors = env.time_based_price_probs
    high_price_probs = []

    for t in range(12):
        # 高电价概率由供电侧策略和时间因子共同决定
        base_prob = supply_final_action[0]  # 高电价基础概率
        time_factor = time_factors[t]
        # 计算最终概率
        prob = base_prob * time_factor * 0.8 + 0.2 * time_factor
        prob = np.clip(prob, 0.05, 0.95)
        high_price_probs.append(round(prob, 3))

    return high_price_probs, supply_final_action, load_final_action


# 运行博弈模型训练并获取高电价概率
DAY_AHEAD_HIGH_PRICE_PROB, supply_strategy, load_strategy = train_game_theory_model()
print("博弈模型生成的高电价概率分布：", DAY_AHEAD_HIGH_PRICE_PROB)
print("供电侧均衡策略（高、中、低）：", supply_strategy)
print("负载侧均衡策略（高、中、低）：", load_strategy)

# ==================== 增强参数设定 ====================
P_d_base = [12, 10, 12, 18, 25, 30, 32, 30, 28, 25, 20, 15]
gen_P = [13.5, 10.75, 6.83, 5.34, 2.87, 8.7, 10.9]  # 7台机组
gen_cost = [80, 90, 110, 105, 120, 95, 92]
N_GEN = 7
HORIZON = 12
MAX_EPISODES = 6000
BUFFER_SIZE = 100000
BATCH_SIZE = 256
GAMMA = 0.99
LR_ACTOR = 2e-4
LR_CRITIC = 6e-4
LR_WORLD_MODEL = 8e-4
TAU = 0.005
UNCERTAINTY_RADIUS = 2.5  # 降低不确定性半径
PRICE_SENSITIVITY = 0.4  # 提高价格敏感度以增强负载响应
DEMAND_VARIABILITY = 0.25  # 提高需求变异性
ROBUST_RADIUS = 0.05
GRADIENT_PENALTY_WEIGHT = 8.0
VARIANCE_PENALTY_WEIGHT = 0.003  # 降低方差惩罚权重，允许更大波动

STARTUP_COST = 180.0
MIN_ON_TIME = [2, 3, 2, 4, 2, 3, 3]
MIN_OFF_TIME = [1, 2, 1, 3, 1, 2, 2]

# 使用博弈模型生成的高电价概率
print("本次运行高电价概率分布（由博弈模型生成）：", DAY_AHEAD_HIGH_PRICE_PROB)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ==================== 增强环境 - 改进负载侧波动 ====================
class RealisticVPPEnv:
    def __init__(self):
        self.gen_cost = gen_cost
        self.gen_P = gen_P
        self.min_on_time = MIN_ON_TIME
        self.min_off_time = MIN_OFF_TIME
        self.startup_cost = STARTUP_COST

        # 从博弈模型获取策略
        self.supply_strategy = supply_strategy
        self.load_strategy = load_strategy

        self.reset()

    def reset(self):
        self.t = 0
        self.prev_on = np.zeros(N_GEN, dtype=np.float32)
        self.remain_min_on = np.array(self.min_on_time, dtype=np.float32)
        self.remain_min_off = np.zeros(N_GEN, dtype=np.float32)

        # 历史数据
        self.demand_history = []
        self.gen_history = []
        self.gap_history = []
        self.price_history = []
        self.load_ratio_history = []

        # 负载侧特性 - 增大波动幅度，更真实
        self.load_inertia = np.random.uniform(0.4, 0.6)  # 负载惯性，适当降低以增加灵活性
        self.load_max_change = 0.4  # 增大到±40%，允许更大波动
        self.load_preference_profile = np.random.uniform(-0.2, 0.2, HORIZON)  # 用户偏好模式，增加幅度
        self.load_response_delay = np.random.randint(1, 3)  # 响应延迟(1-2个时段)

        # 需求侧管理参数
        self.elasticity = np.random.uniform(0.2, 0.4)  # 价格弹性，提高以增强响应
        self.comfort_factor = np.random.uniform(0.6, 0.8)  # 舒适度因子，降低以允许更大调整
        self.load_pattern = np.random.uniform(-0.15, 0.15, HORIZON)  # 日负荷模式，增加幅度

        # 需求不确定性
        self.weather_effect = np.random.normal(0, 0.08)  # 天气影响，增加幅度
        self.random_event_prob = 0.08  # 随机事件概率，提高以增加不确定性

        return self.get_state()

    def get_state(self):
        onehot = np.zeros(HORIZON, dtype=np.float32)
        if self.t < HORIZON:
            onehot[self.t] = 1.0

        # 历史需求特征
        if len(self.demand_history) > 0:
            recent_demand = self.demand_history[-min(5, len(self.demand_history)):]
            demand_avg = np.mean(recent_demand) if recent_demand else 0
            demand_std = np.std(recent_demand) if len(recent_demand) > 1 else 0
            demand_trend = np.polyfit(range(len(recent_demand)), recent_demand, 1)[0] if len(recent_demand) >= 2 else 0
        else:
            demand_avg = 0
            demand_std = 0
            demand_trend = 0

        # 负载调整特征
        if len(self.load_ratio_history) > 0:
            recent_load_adj = self.load_ratio_history[-min(3, len(self.load_ratio_history)):]
            load_adj_avg = np.mean(recent_load_adj) if recent_load_adj else 0
            load_adj_std = np.std(recent_load_adj) if len(recent_load_adj) > 1 else 0
        else:
            load_adj_avg = 0
            load_adj_std = 0

        # 电价概率特征（使用博弈模型生成的概率）
        if self.t < len(DAY_AHEAD_HIGH_PRICE_PROB):
            current_price_prob = DAY_AHEAD_HIGH_PRICE_PROB[self.t]
            future_price_avg = np.mean(DAY_AHEAD_HIGH_PRICE_PROB[min(self.t + 1, 11):]) if self.t < 11 else \
            DAY_AHEAD_HIGH_PRICE_PROB[self.t]
        else:
            current_price_prob = 0
            future_price_avg = 0

        # 博弈策略特征（添加博弈策略信息）
        supply_strategy_feature = self.supply_strategy
        load_strategy_feature = self.load_strategy

        # 时间特征
        hour_feature = np.sin(2 * np.pi * self.t / 24)
        hour_feature2 = np.cos(2 * np.pi * self.t / 24)

        # 构建状态向量（增加了博弈策略特征）
        state = np.concatenate([
            [self.t / HORIZON],
            self.prev_on,
            self.remain_min_on / 5.0,
            self.remain_min_off / 5.0,
            np.array(DAY_AHEAD_HIGH_PRICE_PROB),
            onehot,
            [demand_avg / 50.0, demand_std / 20.0, demand_trend / 10.0],
            [load_adj_avg, load_adj_std, self.load_inertia],
            [current_price_prob, future_price_avg, self.elasticity, self.comfort_factor],
            [hour_feature, hour_feature2, self.weather_effect],
            self.load_preference_profile,
            [self.load_max_change],
            supply_strategy_feature,
            load_strategy_feature
        ])
        return state.astype(np.float32)

    def calculate_realistic_demand(self, load_ratio_action):
        """改进的需求计算：增加负载侧波动幅度"""
        # 1. 限制负载调整幅度和变化率，但允许更大波动
        load_ratio = np.clip(load_ratio_action, -1.0, 1.0) * self.load_max_change

        # 2. 考虑负载惯性，但降低惯性系数以允许更快变化
        if len(self.load_ratio_history) > 0:
            last_load_ratio = self.load_ratio_history[-1]
            # 惯性调整：新调整 = 惯性部分 + 动作部分
            load_ratio = self.load_inertia * last_load_ratio + (1 - self.load_inertia) * load_ratio

        # 3. 基础需求
        base = P_d_base[self.t]

        # 4. 价格响应（增加价格敏感性，使用博弈模型生成的概率）
        price_prob = DAY_AHEAD_HIGH_PRICE_PROB[self.t]
        price_response = -price_prob * self.elasticity * base

        # 5. 博弈策略影响（考虑负载侧均衡策略）
        # 高电价时段，如果负载侧策略倾向于减少用电，则进一步降低需求
        if price_prob > 0.5 and self.load_strategy[0] < 0.3:  # 高电价且负载侧减少高电价时段用电
            price_response *= 1.2  # 增强响应

        # 6. 时间模式（峰谷平特性），增强峰谷差异
        hour = self.t % 24
        if 8 <= hour <= 11 or 18 <= hour <= 21:
            time_factor = 0.2  # 高峰时段，提高系数
        elif 12 <= hour <= 17:
            time_factor = 0.1  # 平时段
        else:
            time_factor = -0.15  # 低谷时段，提高系数

        time_effect = time_factor * base

        # 7. 用户偏好模式，增加幅度
        preference_effect = self.load_preference_profile[self.t] * base * 0.3

        # 8. 天气影响，增加幅度
        weather_effect = self.weather_effect * base * 0.5

        # 9. 需求不确定性，增加幅度以模拟真实波动
        uncertainty = np.random.normal(0, UNCERTAINTY_RADIUS * 0.8)
        uncertainty = np.clip(uncertainty, -UNCERTAINTY_RADIUS * 1.5, UNCERTAINTY_RADIUS * 1.5)

        # 10. 随机事件（设备故障、突发事件等），增加概率和幅度
        if random.random() < self.random_event_prob:
            event_effect = random.uniform(-0.3, 0.3) * base
        else:
            event_effect = 0

        # 11. 相邻时段相关性（需求平滑性），降低相关性以允许更大变化
        if len(self.demand_history) > 0:
            demand_inertia = 0.2 * (self.demand_history[-1] - base)
        else:
            demand_inertia = 0

        # 12. 舒适度约束（负载调整不能影响用户体验），放松约束
        comfort_penalty = 0
        if abs(load_ratio) > 0.3 * self.comfort_factor:  # 放松约束
            comfort_penalty = -abs(load_ratio) * 30  # 降低惩罚

        # 13. 增加周期性波动（模拟工厂、商场等周期性用电）
        periodic_effect = np.sin(2 * np.pi * self.t / 6) * base * 0.1

        # 组合所有因素
        demand = (
                base * (1 + load_ratio)
                + price_response
                + time_effect
                + preference_effect
                + weather_effect
                + uncertainty
                + event_effect
                + demand_inertia
                + periodic_effect
        )

        # 物理约束
        demand = np.clip(demand, 5.0, 55.0)  # 增加上限

        # 最小变化约束（避免微小波动，但允许合理变化）
        if len(self.demand_history) > 0:
            min_change = 0.1 * base  # 增加最小变化
            if abs(demand - self.demand_history[-1]) < min_change:
                # 保持趋势方向，但确保最小变化
                sign = 1 if demand > self.demand_history[-1] else -1
                demand = self.demand_history[-1] + sign * min_change

        return demand, load_ratio, comfort_penalty

    def step(self, load_ratio_raw, supply_raw):
        # 计算需求（使用改进的方法）
        demand, actual_load_ratio, comfort_penalty = self.calculate_realistic_demand(load_ratio_raw)

        # 供应侧调度（保持不变）
        supply = supply_raw.copy()
        startup_cost = running_cost = 0.0

        for i in range(N_GEN):
            if self.remain_min_off[i] > 0:
                supply[i] = 0.0
                self.remain_min_off[i] = max(self.remain_min_off[i] - 1, 0)

            if supply[i] > 0.5 and self.prev_on[i] < 0.5:
                if self.remain_min_off[i] > 0:
                    supply[i] = 0.0
                else:
                    startup_cost += self.startup_cost * (1 + 0.1 * random.random())

            if self.prev_on[i] > 0.5 and supply[i] < 0.5:
                if self.remain_min_on[i] > 0:
                    supply[i] = 1.0

            if supply[i] > 0.5:
                self.remain_min_on[i] = max(self.remain_min_on[i] - 1, 0)
                if self.prev_on[i] < 0.5:
                    self.remain_min_off[i] = 0
                running_cost += self.gen_cost[i] * 0.05 * (self.gen_P[i] / max(self.gen_P))
            else:
                if self.prev_on[i] > 0.5:
                    self.remain_min_off[i] = self.min_off_time[i]
                self.remain_min_on[i] = self.min_on_time[i]

        gen_power = np.sum(supply * np.array(self.gen_P))

        current_prev_on = self.prev_on.copy()
        self.prev_on = supply.copy()

        gap = gen_power - demand
        abs_gap = abs(gap)

        # 开关惩罚
        switch_penalty = 0
        for i in range(N_GEN):
            if supply[i] > 0.5 and current_prev_on[i] < 0.5:
                switch_penalty += self.startup_cost * (0.7 + 0.3 * (self.gen_P[i] / max(self.gen_P)))
            elif supply[i] < 0.5 and current_prev_on[i] > 0.5:
                switch_penalty += 50

        # 奖励计算（调整奖励范围到0到-3000）
        penalty_breakdown = {
            "shortage_penalty": 0,
            "surplus_penalty": 0,
            "gap_penalty": 0,
            "startup_cost": startup_cost,
            "running_cost": running_cost,
            "switch_penalty": switch_penalty,
            "waste_penalty": 0,
            "variance_penalty": 0,
            "load_change_penalty": 0,
            "comfort_penalty": comfort_penalty
        }

        # 缺电惩罚（大幅降低惩罚系数）
        if gap < 0:
            shortage = -gap
            penalty_breakdown["shortage_penalty"] = -400 * shortage - 20 * shortage ** 2

        # 过剩惩罚（大幅降低惩罚系数）
        if gap > 0:
            surplus = gap
            if surplus <= 0.5:
                penalty_breakdown["surplus_penalty"] = -2 * surplus
            elif surplus <= 2.0:
                penalty_breakdown["surplus_penalty"] = -10 * surplus
            else:
                penalty_breakdown["surplus_penalty"] = -50 * surplus - 15 * surplus ** 2

        # 基本偏差惩罚（降低惩罚系数）
        penalty_breakdown["gap_penalty"] = -30 * abs_gap

        # 浪费惩罚（降低惩罚系数）
        if gap > 0:
            penalty_breakdown["waste_penalty"] = -80 * max(0, gap - 0.5)

        # 负载变化惩罚（降低惩罚系数以允许更大变化）
        if len(self.load_ratio_history) > 0:
            load_change = abs(actual_load_ratio - self.load_ratio_history[-1])
            penalty_breakdown["load_change_penalty"] = -5 * load_change ** 2

        # 方差惩罚（降低惩罚系数以允许更大波动）
        action_variance = np.var(supply) + np.var([actual_load_ratio])
        penalty_breakdown["variance_penalty"] = -VARIANCE_PENALTY_WEIGHT * action_variance * 200

        # 总奖励
        total_reward = sum(penalty_breakdown.values())

        # 更新历史记录
        self.demand_history.append(demand)
        self.gen_history.append(gen_power)
        self.gap_history.append(gap)
        self.price_history.append(DAY_AHEAD_HIGH_PRICE_PROB[self.t] if self.t < len(DAY_AHEAD_HIGH_PRICE_PROB) else 0)
        self.load_ratio_history.append(actual_load_ratio)

        self.t += 1
        done = self.t >= HORIZON

        return self.get_state(), float(total_reward), done, {
            "demand": round(demand, 2),
            "gen": round(gen_power, 2),
            "actual_on": supply.copy(),
            "load_ratio": round(actual_load_ratio, 3),
            "gap": round(gap, 2),
            "abs_gap": round(abs_gap, 2),
            "penalty_breakdown": penalty_breakdown,
            "is_shortage": gap < -0.5,
            "is_surplus": gap > 0.5,
            "is_balanced": abs_gap <= 0.5,
            "startup_cost": startup_cost,
            "running_cost": running_cost,
            "switch_penalty": switch_penalty,
            "action_variance": action_variance,
            "comfort_penalty": comfort_penalty,
            "load_change_penalty": penalty_breakdown["load_change_penalty"],
            "price_prob": DAY_AHEAD_HIGH_PRICE_PROB[self.t - 1] if self.t > 0 else 0
        }


# ==================== 概率世界模型 (完整版) ====================
class VRNNCell(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(64, z_dim)
        self.enc_logvar = nn.Linear(64, z_dim)

        self.prior = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(64, z_dim)
        self.prior_logvar = nn.Linear(64, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.dec_mean = nn.Linear(64, x_dim)
        self.dec_logvar = nn.Linear(64, x_dim)

        self.rnn = nn.GRUCell(x_dim + z_dim, h_dim)

    def forward(self, x, h):
        prior_hidden = self.prior(h)
        prior_mean = self.prior_mean(prior_hidden)
        prior_logvar = self.prior_logvar(prior_hidden)

        enc_hidden = self.enc(torch.cat([x, h], dim=1))
        enc_mean = self.enc_mean(enc_hidden)
        enc_logvar = self.enc_logvar(enc_hidden)

        eps = torch.randn_like(enc_mean) * 0.1
        z = enc_mean + torch.exp(0.5 * enc_logvar) * eps

        dec_hidden = self.dec(torch.cat([z, h], dim=1))
        dec_mean = self.dec_mean(dec_hidden)
        dec_logvar = self.dec_logvar(dec_hidden)

        h_next = self.rnn(torch.cat([x, z], dim=1), h)

        return dec_mean, dec_logvar, enc_mean, enc_logvar, prior_mean, prior_logvar, h_next, z


class ProbabilisticWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=128, z_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.input_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.vrnn = VRNNCell(128, h_dim, z_dim)

        self.output_decoder = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action, h):
        x = torch.cat([state, action], dim=1)
        x_encoded = self.input_encoder(x)

        dec_mean, dec_logvar, enc_mean, enc_logvar, prior_mean, prior_logvar, h_next, z = self.vrnn(x_encoded, h)

        next_state_pred = self.output_decoder(h_next)
        reward_pred = self.reward_predictor(h_next)

        return {
            'next_state_pred': next_state_pred,
            'reward_pred': reward_pred,
            'dec_mean': dec_mean,
            'dec_logvar': dec_logvar,
            'enc_mean': enc_mean,
            'enc_logvar': enc_logvar,
            'prior_mean': prior_mean,
            'prior_logvar': prior_logvar,
            'h_next': h_next,
            'z': z
        }

    def compute_loss(self, state, action, next_state, reward, h):
        output = self.forward(state, action, h)

        recon_loss = F.mse_loss(output['next_state_pred'], next_state)

        kl_div = -0.5 * torch.sum(1 + output['enc_logvar'] - output['prior_logvar']
                                  - (output['enc_mean'] - output['prior_mean']).pow(2) / (
                                          output['prior_logvar'].exp() + 1e-8)
                                  - output['enc_logvar'].exp() / (output['prior_logvar'].exp() + 1e-8), dim=1).mean()

        reward_pred = output['reward_pred']
        reward_target = reward.unsqueeze(1) if reward.dim() == 1 else reward
        if reward_pred.shape != reward_target.shape:
            reward_pred = reward_pred.squeeze(-1)
            reward_target = reward_target.squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, reward_target)

        total_loss = recon_loss + 0.1 * kl_div + 0.1 * reward_loss

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'kl_div': kl_div.item(),
            'reward_loss': reward_loss.item()
        }


# ==================== 改进的负载Actor（考虑惯性，允许更大波动） ====================
class InertialLoadActor(nn.Module):
    def __init__(self, s_dim, noise_scale=0.2, dropout_prob=0.1):
        super().__init__()
        self.s_dim = s_dim
        self.noise_scale = noise_scale
        self.dropout_prob = dropout_prob

        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 负载惯性学习分支
        self.inertia_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出惯性系数[0,1]
        )

        # 动作生成分支（增加输出范围）
        self.action_net = nn.Sequential(
            nn.Linear(128 + 1, 64),  # +1 for inertia coefficient
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # 不确定性估计
        self.uncertainty_net = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x, last_action=None, deterministic=False, training=True):
        # 确保输入数据类型正确
        if x.dtype != torch.float32:
            x = x.float()

        if training and self.training:
            # 增加噪声以增强探索
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise

        features = self.feature_extractor(x)

        # 学习惯性系数
        inertia = self.inertia_net(features)

        # 如果提供了上一时刻动作，考虑惯性
        if last_action is not None:
            # 确保last_action在相同设备上并且数据类型正确
            if last_action.device != x.device:
                last_action = last_action.to(x.device)
            if last_action.dtype != torch.float32:
                last_action = last_action.float()

            inertia_features = torch.cat([features, inertia], dim=1)
            raw_action = self.action_net(inertia_features)

            # 应用惯性：新动作 = 惯性×旧动作 + (1-惯性)×新动作
            action = inertia * last_action + (1 - inertia) * raw_action
        else:
            inertia_features = torch.cat([features, inertia], dim=1)
            action = self.action_net(inertia_features)

        # 估计不确定性
        uncertainty = self.uncertainty_net(features)

        # 扩大动作范围以允许更大波动
        if deterministic or not training:
            return torch.clamp(action, -1.0, 1.0), uncertainty, inertia
        else:
            # 增加探索噪声
            exploration_noise = torch.randn_like(action) * 0.15  # 增加噪声
            action = torch.clamp(action + exploration_noise, -1.0, 1.0)
            return action, uncertainty, inertia


# ==================== 不确定性感知供应Actor ====================
class UncertaintyAwareSupplyActor(nn.Module):
    def __init__(self, s_dim, noise_scale=0.2, dropout_prob=0.1):
        super().__init__()
        self.s_dim = s_dim
        self.noise_scale = noise_scale
        self.dropout_prob = dropout_prob

        self.net = nn.Sequential(
            nn.Linear(s_dim + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, N_GEN * 2)
        )

    def forward(self, x, training=True, num_samples=1):
        # 确保输入数据类型正确
        if x.dtype != torch.float32:
            x = x.float()

        if training and self.training:
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise

        logits = self.net(x).view(-1, N_GEN, 2)

        return logits, torch.zeros(x.size(0), 1, device=x.device, dtype=torch.float32)


# ==================== 分布鲁棒Critic ====================
class DistributionRobustCritic(nn.Module):
    def __init__(self, s_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + N_GEN + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        # 确保输入数据类型正确
        if state.dtype != torch.float32:
            state = state.float()
        if action.dtype != torch.float32:
            action = action.float()

        x = torch.cat([state, action], dim=1)
        return self.net(x)

    def compute_gradient_penalty(self, real_samples, generated_samples, state):
        # 确保输入数据类型正确
        if real_samples.dtype != torch.float32:
            real_samples = real_samples.float()
        if generated_samples.dtype != torch.float32:
            generated_samples = generated_samples.float()
        if state.dtype != torch.float32:
            state = state.float()

        alpha = torch.rand(real_samples.size(0), 1).to(real_samples.device)
        interpolated = alpha * real_samples + (1 - alpha) * generated_samples

        interpolated.requires_grad_(True)
        critic_interpolated = self.forward(state, interpolated)

        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()

        return gradient_penalty


# ==================== 完整的DROMARL算法 ====================
class DROMARL:
    def __init__(self, s_dim):
        self.s_dim = s_dim

        # 主要网络
        self.load_actor = InertialLoadActor(s_dim).to(device)
        self.supply_actor = UncertaintyAwareSupplyActor(s_dim).to(device)
        self.critic = DistributionRobustCritic(s_dim).to(device)
        self.world_model = ProbabilisticWorldModel(s_dim, N_GEN + 1, h_dim=128, z_dim=32).to(device)

        # 目标网络
        self.load_actor_target = copy.deepcopy(self.load_actor)
        self.supply_actor_target = copy.deepcopy(self.supply_actor)
        self.critic_target = copy.deepcopy(self.critic)

        # 优化器
        self.opt_load = optim.Adam(self.load_actor.parameters(), lr=LR_ACTOR)
        self.opt_supply = optim.Adam(self.supply_actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.opt_world_model = optim.Adam(self.world_model.parameters(), lr=LR_WORLD_MODEL)

        # 经验回放
        self.buffer = deque(maxlen=BUFFER_SIZE)

        # 训练参数
        self.temperature = 1.0
        self.noise_scale = 0.25  # 增加噪声尺度以增强探索
        self.uncertainty_threshold = 0.8  # 提高不确定性阈值

        # 记录
        self.critic_losses = []
        self.actor_losses = []
        self.world_model_losses = []
        self.uncertainty_history = []
        self.load_inertia_history = []

    def select_action(self, state, last_load_action=None, deterministic=False, evaluate=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)

        # 确保last_load_action在正确设备上并且数据类型正确
        if last_load_action is not None:
            if isinstance(last_load_action, torch.Tensor):
                last_load_action = last_load_action.to(device).float()
            else:
                last_load_action = torch.FloatTensor(last_load_action).unsqueeze(0).to(device)

        with torch.no_grad():
            # 负载侧动作
            if evaluate:
                load_action, _, inertia = self.load_actor(s, last_load_action, deterministic=True, training=False)
            else:
                load_action, _, inertia = self.load_actor(s, last_load_action, deterministic=False, training=True)

            # 确保load_action数据类型正确
            load_action = load_action.float()

            # 供应侧动作
            s_with_load = torch.cat([s, load_action], dim=1).float()

            if evaluate or deterministic:
                supply_logits, _ = self.supply_actor(s_with_load, training=False)
                supply = (supply_logits[:, :, 1] > supply_logits[:, :, 0]).float().squeeze(0)
            else:
                supply_logits, _ = self.supply_actor(s_with_load, training=True)
                prob = F.gumbel_softmax(supply_logits, tau=self.temperature, hard=False)
                supply = prob[:, :, 1].squeeze(0)

            action = torch.cat([supply, load_action.squeeze(0)], dim=0)
            return action.cpu().numpy(), inertia.item() if inertia is not None else 0.5

    def store(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def update_world_model(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0, {}

        batch = random.sample(self.buffer, BATCH_SIZE)
        s_batch, a_batch, r_batch, s_next_batch, done_batch = zip(*batch)

        s = torch.FloatTensor(np.stack(s_batch)).to(device)
        a = torch.FloatTensor(np.stack(a_batch)).to(device)
        r = torch.FloatTensor(r_batch).to(device)
        s_next = torch.FloatTensor(np.stack(s_next_batch)).to(device)

        h = torch.zeros(s.size(0), self.world_model.h_dim, device=device, dtype=torch.float32)

        self.opt_world_model.zero_grad()
        world_model_loss, loss_details = self.world_model.compute_loss(s, a, s_next, r, h)

        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        world_model_loss.backward()
        self.opt_world_model.step()

        self.world_model_losses.append(world_model_loss.item())
        return world_model_loss.item(), loss_details

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0, 0, 0

        batch = random.sample(self.buffer, BATCH_SIZE)
        s_batch, a_batch, r_batch, s_next_batch, done_batch = zip(*batch)

        s = torch.FloatTensor(np.stack(s_batch)).to(device)
        a = torch.FloatTensor(np.stack(a_batch)).to(device)
        r = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
        s_next = torch.FloatTensor(np.stack(s_next_batch)).to(device)
        done = torch.FloatTensor(done_batch).unsqueeze(1).to(device)

        # 更新Critic
        current_q = self.critic(s, a)

        with torch.no_grad():
            # 获取负载侧上一动作（从历史中提取）
            last_load_actions = a[:, -1].unsqueeze(1).to(device)

            # 目标动作
            next_load, _, _ = self.load_actor_target(s_next, last_load_actions, deterministic=True, training=False)
            next_load = next_load.float()

            s_next_aug = torch.cat([s_next, next_load], dim=1).float()
            next_logits, _ = self.supply_actor_target(s_next_aug, training=False)
            next_supply_prob = F.softmax(next_logits / 0.1, dim=-1)[:, :, 1]
            next_action = torch.cat([next_supply_prob, next_load], dim=1)

            target_q = self.critic_target(s_next, next_action)
            y = r + GAMMA * (1 - done) * target_q

        # 分布鲁棒Critic训练
        with torch.no_grad():
            noise = torch.randn_like(a) * ROBUST_RADIUS
            a_perturbed = torch.clamp(a + noise, -1.0, 1.0)

        gradient_penalty = self.critic.compute_gradient_penalty(a, a_perturbed, s)

        critic_loss = F.mse_loss(current_q, y) + GRADIENT_PENALTY_WEIGHT * gradient_penalty

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # 更新Actor
        # 负载Actor更新
        load_action, load_uncertainty, inertia = self.load_actor(s, None, deterministic=False, training=True)
        load_action = load_action.float()
        s_aug = torch.cat([s, load_action], dim=1).float()
        supply_logits, supply_uncertainty = self.supply_actor(s_aug, training=True)
        supply_prob = F.gumbel_softmax(supply_logits, tau=self.temperature, hard=False)[:, :, 1]
        predicted_action = torch.cat([supply_prob, load_action], dim=1)

        actor_loss_load = -self.critic(s, predicted_action).mean()

        # 添加不确定性惩罚
        if load_uncertainty.mean() > self.uncertainty_threshold:
            actor_loss_load += 0.02 * load_uncertainty.mean()  # 降低不确定性惩罚

        self.opt_load.zero_grad()
        actor_loss_load.backward()
        torch.nn.utils.clip_grad_norm_(self.load_actor.parameters(), 0.5)
        self.opt_load.step()

        # 供应Actor更新
        with torch.no_grad():
            best_load, _, _ = self.load_actor(s, None, deterministic=True, training=False)
            best_load = best_load.float()

        s_best = torch.cat([s, best_load], dim=1).float()
        supply_logits2, supply_uncertainty2 = self.supply_actor(s_best, training=True)
        supply_prob2 = F.gumbel_softmax(supply_logits2, tau=self.temperature, hard=False)[:, :, 1]
        final_action = torch.cat([supply_prob2, best_load], dim=1)

        actor_loss_supply = -self.critic(s, final_action).mean()

        # 添加方差惩罚（降低权重以允许更大波动）
        action_variance = torch.var(final_action, dim=1).mean()
        actor_loss_supply += VARIANCE_PENALTY_WEIGHT * action_variance

        self.opt_supply.zero_grad()
        actor_loss_supply.backward()
        torch.nn.utils.clip_grad_norm_(self.supply_actor.parameters(), 0.5)
        self.opt_supply.step()

        # 软更新目标网络
        for p, tp in zip(self.load_actor.parameters(), self.load_actor_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        for p, tp in zip(self.supply_actor.parameters(), self.supply_actor_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        # 衰减参数
        self.temperature = max(0.3, self.temperature * 0.9998)
        self.noise_scale = max(0.08, self.noise_scale * 0.9995)  # 减缓衰减以保持探索

        # 记录
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append((actor_loss_load.item() + actor_loss_supply.item()) / 2)
        self.load_inertia_history.append(inertia.mean().item())

        return critic_loss.item(), (actor_loss_load.item() + actor_loss_supply.item()) / 2, 0.0


# ==================== 主训练循环 ====================
env = RealisticVPPEnv()
s_dim = len(env.get_state())
print(f"状态维度: {s_dim}")
agent = DROMARL(s_dim)

print("\n" + "=" * 120)
print("【完整DROMARL训练】基于分布鲁棒多智能体强化学习的VPP优化调度系统")
print("特征：1)调整后的奖励函数(0到-3000) 2)增强的负载波动(±40%) 3)概率世界模型 4)分布鲁棒Critic 5)博弈模型集成")
print("=" * 120)

episode_rewards = []
episode_demands = []
episode_gens = []
episode_gaps = []
balance_stats = []
load_adjustments = []
comfort_violations = []

for ep in range(1, MAX_EPISODES + 1):
    s = env.reset()
    ep_reward = 0
    ep_demands = []
    ep_gens = []
    ep_gaps = []
    ep_load_adjustments = []
    ep_comfort_violations = 0
    balance_count = 0

    last_load_action = None

    while True:
        a, inertia = agent.select_action(s, last_load_action, deterministic=False)
        ns, r, done, info = env.step(a[7], a[:7])
        agent.store(s, a, r, ns, float(done))

        # 更新世界模型
        world_model_loss, wm_loss_details = agent.update_world_model()

        # 更新Actor和Critic
        critic_loss, actor_loss, uncertainty = agent.update()

        ep_reward += r
        ep_demands.append(info["demand"])
        ep_gens.append(info["gen"])
        ep_gaps.append(info["abs_gap"])
        ep_load_adjustments.append(info["load_ratio"])

        if info["comfort_penalty"] < 0:
            ep_comfort_violations += 1

        if info["is_balanced"]:
            balance_count += 1

        # 更新上一时刻动作
        if last_load_action is None:
            last_load_action = torch.tensor([[info["load_ratio"]]], dtype=torch.float32)
        else:
            last_load_action = torch.tensor([[info["load_ratio"]]], dtype=torch.float32)

        s = ns
        if done:
            break

    episode_rewards.append(ep_reward)
    episode_demands.append(np.mean(ep_demands))
    episode_gens.append(np.mean(ep_gens))
    episode_gaps.append(np.mean(ep_gaps))
    balance_stats.append(balance_count / HORIZON * 100)
    load_adjustments.append(np.mean(np.abs(ep_load_adjustments)))
    comfort_violations.append(ep_comfort_violations / HORIZON * 100)

    if ep % 200 == 0:
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_balance = np.mean(balance_stats[-100:]) if len(balance_stats) >= 100 else np.mean(balance_stats)
        avg_gap = np.mean(episode_gaps[-100:]) if len(episode_gaps) >= 100 else np.mean(episode_gaps)
        avg_load_adj = np.mean(load_adjustments[-100:]) if len(load_adjustments) >= 100 else np.mean(load_adjustments)
        avg_comfort_violation = np.mean(comfort_violations[-100:]) if len(comfort_violations) >= 100 else np.mean(
            comfort_violations)
        print(f"Episode {ep:4d} | 奖励: {avg_reward:+.1f} | 平衡率: {avg_balance:.1f}% | "
              f"偏差: {avg_gap:.2f}kW | 负载调整: {avg_load_adj:.3f} | 舒适违规: {avg_comfort_violation:.1f}%")

# ==================== 性能分析 ====================
print("\n" + "=" * 120)
print("训练完成！完整DROMARL性能分析报告")
print("=" * 120)

window = min(100, len(episode_rewards))
final_demands = episode_demands[-window:] if len(episode_demands) >= window else episode_demands
final_gens = episode_gens[-window:] if len(episode_gens) >= window else episode_gens
final_gaps = episode_gaps[-window:] if len(episode_gaps) >= window else episode_gaps
final_balance = balance_stats[-window:] if len(balance_stats) >= window else balance_stats
final_load_adj = load_adjustments[-window:] if len(load_adjustments) >= window else load_adjustments
final_comfort = comfort_violations[-window:] if len(comfort_violations) >= window else comfort_violations

print(f"【供需平衡统计】")
print(f"  - 平均发电功率: {np.mean(final_gens):.2f} ± {np.std(final_gens):.2f} kW")
print(f"  - 平均需求功率: {np.mean(final_demands):.2f} ± {np.std(final_demands):.2f} kW")
print(f"  - 平均绝对偏差: {np.mean(final_gaps):.2f} kW")
print(f"  - 供需平衡率: {np.mean(final_balance):.1f}%")
print(f"  - 平均负载调整幅度: {np.mean(final_load_adj):.3f}")
print(f"  - 舒适度违规率: {np.mean(final_comfort):.1f}%")
print(f"  - 平均奖励范围: {np.mean(episode_rewards[-window:]):.1f} (目标: 0到-3000)")

# 计算需求变化特征
demand_changes = []
load_adj_changes = []
for i in range(1, window):
    if i < len(final_demands):
        demand_changes.append(abs(final_demands[i] - final_demands[i - 1]))
    if i < len(final_load_adj):
        load_adj_changes.append(abs(final_load_adj[i] - final_load_adj[i - 1]))

print(f"  - 平均需求变化: {np.mean(demand_changes):.2f} kW")
print(f"  - 平均负载调整变化: {np.mean(load_adj_changes):.3f}")
print(f"  - 平均负载惯性系数: {np.mean(agent.load_inertia_history[-100:]):.3f}")

# ==================== 可视化结果 ====================
fig = plt.figure(figsize=(24, 20))

# 1. 奖励收敛曲线
ax1 = plt.subplot(4, 4, 1)
ax1.plot(episode_rewards, alpha=0.6, color='lightblue', label='单轮奖励')
if len(episode_rewards) > 100:
    smoothed = np.convolve(episode_rewards, np.ones(100) / 100, mode='valid')
    ax1.plot(range(99, len(episode_rewards)), smoothed, color='red', linewidth=2, label='滑动平均')
ax1.axhline(y=-3000, color='red', linestyle='--', alpha=0.5, label='目标上限(-3000)')
ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='理论最优(0)')
ax1.set_title('奖励收敛曲线 (调整后)', fontsize=12, fontweight='bold')
ax1.set_xlabel('训练轮数')
ax1.set_ylabel('累计奖励')
ax1.grid(alpha=0.3)
ax1.legend()

# 2. 供需平衡率趋势
ax2 = plt.subplot(4, 4, 2)
ax2.plot(balance_stats, alpha=0.6, color='green')
if len(balance_stats) > 100:
    smoothed_balance = np.convolve(balance_stats, np.ones(100) / 100, mode='valid')
    ax2.plot(range(99, len(balance_stats)), smoothed_balance, color='darkgreen', linewidth=2)
ax2.set_title('供需平衡率趋势', fontsize=12, fontweight='bold')
ax2.set_xlabel('训练轮数')
ax2.set_ylabel('平衡率 (%)')
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 105)

# 3. 负载调整幅度趋势
ax3 = plt.subplot(4, 4, 3)
ax3.plot(load_adjustments, alpha=0.6, color='purple')
if len(load_adjustments) > 100:
    smoothed_load = np.convolve(load_adjustments, np.ones(100) / 100, mode='valid')
    ax3.plot(range(99, len(load_adjustments)), smoothed_load, color='darkviolet', linewidth=2)
ax3.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='最大调整(40%)')
ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='舒适度阈值(30%)')
ax3.set_title('负载调整幅度趋势 (增强后)', fontsize=12, fontweight='bold')
ax3.set_xlabel('训练轮数')
ax3.set_ylabel('负载调整幅度')
ax3.grid(alpha=0.3)
ax3.legend()

# 4. 舒适度违规率
ax4 = plt.subplot(4, 4, 4)
ax4.plot(comfort_violations, alpha=0.6, color='orange')
if len(comfort_violations) > 100:
    smoothed_comfort = np.convolve(comfort_violations, np.ones(100) / 100, mode='valid')
    ax4.plot(range(99, len(comfort_violations)), smoothed_comfort, color='darkorange', linewidth=2)
ax4.set_title('舒适度违规率趋势', fontsize=12, fontweight='bold')
ax4.set_xlabel('训练轮数')
ax4.set_ylabel('违规率 (%)')
ax4.grid(alpha=0.3)
ax4.set_ylim(0, 50)

# 5. 训练损失曲线
ax5 = plt.subplot(4, 4, 5)
if len(agent.critic_losses) > 0:
    critic_losses_smooth = np.convolve(agent.critic_losses[-min(500, len(agent.critic_losses)):], np.ones(50) / 50,
                                       mode='valid')
    ax5.plot(critic_losses_smooth, color='blue', alpha=0.7, label='Critic损失')
if len(agent.actor_losses) > 0:
    actor_losses_smooth = np.convolve(agent.actor_losses[-min(500, len(agent.actor_losses)):], np.ones(50) / 50,
                                      mode='valid')
    ax5.plot(actor_losses_smooth, color='red', alpha=0.7, label='Actor损失')
if len(agent.world_model_losses) > 0:
    wm_losses_smooth = np.convolve(agent.world_model_losses[-min(500, len(agent.world_model_losses)):],
                                   np.ones(50) / 50, mode='valid')
    ax5.plot(wm_losses_smooth, color='green', alpha=0.7, label='世界模型损失')
ax5.set_title('训练损失曲线', fontsize=12, fontweight='bold')
ax5.set_xlabel('更新步数')
ax5.set_ylabel('损失')
ax5.grid(alpha=0.3)
ax5.legend()

# 6. 最终调度结果
ax6 = plt.subplot(4, 4, (6, 9))
final_demand_curve = []
final_gen_curve = []
final_gap_curve = []
final_load_curve = []
final_price_probs = []

env.reset()
last_load = None
for t in range(12):
    a, _ = agent.select_action(env.get_state(), last_load, deterministic=True, evaluate=True)
    _, _, _, info = env.step(a[7], a[:7])
    final_demand_curve.append(info["demand"])
    final_gen_curve.append(info["gen"])
    final_gap_curve.append(info["gap"])
    final_load_curve.append(info["load_ratio"])
    final_price_probs.append(info["price_prob"])
    last_load = torch.tensor([[info["load_ratio"]]], dtype=torch.float32)

hours = range(1, 13)
ax6.plot(hours, final_demand_curve, 'o-', color='blue', linewidth=3, markersize=8, label='实际需求')
ax6.plot(hours, final_gen_curve, 's-', color='red', linewidth=3, markersize=8, label='发电功率')
ax6.plot(hours, P_d_base, '--', color='gray', linewidth=2, alpha=0.7, label='基础负荷')

ax6.fill_between(hours, final_demand_curve, final_gen_curve,
                 where=np.array(final_gen_curve) >= np.array(final_demand_curve),
                 alpha=0.3, color='green', label='发电过剩')
ax6.fill_between(hours, final_demand_curve, final_gen_curve,
                 where=np.array(final_gen_curve) < np.array(final_demand_curve),
                 alpha=0.3, color='red', label='供电不足')

ax6.set_title('最终调度结果：发电与需求匹配', fontsize=12, fontweight='bold')
ax6.set_xlabel('时间段 (小时)')
ax6.set_ylabel('功率 (kW)')
ax6.grid(alpha=0.3)
ax6.legend()
ax6.set_xticks(hours)

# 7. 负载调整与电价关系
ax7 = plt.subplot(4, 4, 10)
width = 0.35
x = np.arange(12)
ax7.bar(x - width / 2, final_price_probs, width, color='orange', alpha=0.7, label='高电价概率')
ax7.bar(x + width / 2, final_load_curve, width, color='blue', alpha=0.7, label='负载调整率')
ax7.axhline(y=0, color='black', linewidth=0.5)
ax7.set_title('电价概率 vs 负载调整', fontsize=12, fontweight='bold')
ax7.set_xlabel('时间段')
ax7.set_ylabel('概率 / 调整率')
ax7.grid(alpha=0.3, axis='y')
ax7.legend()
ax7.set_xticks(x)

# 8. 需求变化分析
ax8 = plt.subplot(4, 4, 11)
base_vs_actual = [(d - base) / base * 100 for d, base in zip(final_demand_curve, P_d_base)]
ax8.plot(hours, base_vs_actual, 'o-', color='green', linewidth=2, markersize=6)
ax8.fill_between(hours, 0, base_vs_actual, where=np.array(base_vs_actual) > 0,
                 alpha=0.3, color='red', label='需求增加')
ax8.fill_between(hours, 0, base_vs_actual, where=np.array(base_vs_actual) < 0,
                 alpha=0.3, color='blue', label='需求减少')
ax8.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax8.set_title('需求相对变化率 (增强波动)', fontsize=12, fontweight='bold')
ax8.set_xlabel('时间段')
ax8.set_ylabel('变化率 (%)')
ax8.grid(alpha=0.3)
ax8.legend()

# 9. 机组启停策略
ax9 = plt.subplot(4, 4, 12)
env.reset()
all_actions = []
last_load = None
for t in range(12):
    a, _ = agent.select_action(env.get_state(), last_load, deterministic=True, evaluate=True)
    _, _, _, info = env.step(a[7], a[:7])
    all_actions.append(info["actual_on"])
    last_load = torch.tensor([[info["load_ratio"]]], dtype=torch.float32)

im = ax9.imshow(np.array(all_actions).T, cmap='RdYlGn', aspect='auto', interpolation='nearest')
ax9.set_title('机组启停策略', fontsize=12, fontweight='bold')
ax9.set_xlabel('时间段')
ax9.set_ylabel('机组')
ax9.set_yticks(range(N_GEN))
ax9.set_yticklabels([f'G{i + 1}' for i in range(N_GEN)])
plt.colorbar(im, ax=ax9, orientation='vertical', fraction=0.046, pad=0.04)

# 10. 负载调整细节
ax10 = plt.subplot(4, 4, 13)
load_changes = []
for i in range(1, len(final_load_curve)):
    load_changes.append(abs(final_load_curve[i] - final_load_curve[i - 1]))

ax10.bar(range(2, 13), load_changes, color='purple', alpha=0.7)
ax10.axhline(y=np.mean(load_changes), color='red', linestyle='--', label=f'均值: {np.mean(load_changes):.3f}')
ax10.set_title('负载调整变化率', fontsize=12, fontweight='bold')
ax10.set_xlabel('时间段')
ax10.set_ylabel('调整变化')
ax10.grid(alpha=0.3, axis='y')
ax10.legend()

# 11. 需求响应效果
ax11 = plt.subplot(4, 4, 14)
response_effectiveness = []
for i in range(12):
    price = final_price_probs[i]
    load_adj = final_load_curve[i]
    if price > 0.5:  # 高电价时段
        expected = -price * 0.4  # 期望调整（提高期望值）
        actual = load_adj
        effectiveness = 1 - abs(expected - actual) / abs(expected) if expected != 0 else 1
        response_effectiveness.append(max(0, min(1, effectiveness)))
    else:
        response_effectiveness.append(0)

ax11.bar(hours, response_effectiveness, color='teal', alpha=0.7)
ax11.axhline(y=np.mean(response_effectiveness), color='red', linestyle='--',
             label=f'平均响应率: {np.mean(response_effectiveness):.1%}')
ax11.set_title('价格响应有效性', fontsize=12, fontweight='bold')
ax11.set_xlabel('时间段')
ax11.set_ylabel('响应率')
ax11.set_ylim(0, 1.1)
ax11.grid(alpha=0.3, axis='y')
ax11.legend()

# 12. 负载惯性系数趋势
ax12 = plt.subplot(4, 4, 15)
if len(agent.load_inertia_history) > 0:
    ax12.plot(agent.load_inertia_history[-min(500, len(agent.load_inertia_history)):], alpha=0.6, color='brown')
    if len(agent.load_inertia_history) > 100:
        inertia_smooth = np.convolve(agent.load_inertia_history[-min(500, len(agent.load_inertia_history)):],
                                     np.ones(100) / 100, mode='valid')
        ax12.plot(range(99, len(agent.load_inertia_history[-min(500, len(agent.load_inertia_history)):])),
                  inertia_smooth, color='darkred', linewidth=2)
    ax12.axhline(y=np.mean(agent.load_inertia_history[-100:]), color='red', linestyle='--',
                 label=f'平均惯性: {np.mean(agent.load_inertia_history[-100:]):.3f}')
    ax12.set_title('负载惯性系数趋势', fontsize=12, fontweight='bold')
    ax12.set_xlabel('更新步数')
    ax12.set_ylabel('惯性系数')
    ax12.grid(alpha=0.3)
    ax12.legend()

# 13. 需求波动统计
ax13 = plt.subplot(4, 4, 16)
demand_range = np.max(final_demand_curve) - np.min(final_demand_curve)
demand_std = np.std(final_demand_curve)
demand_cv = (demand_std / np.mean(final_demand_curve)) * 100

stats_data = [demand_range, demand_std, demand_cv]
stats_labels = ['需求范围(kW)', '标准差(kW)', '变异系数(%)']
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = ax13.bar(stats_labels, stats_data, color=colors, alpha=0.7)
for bar, val in zip(bars, stats_data):
    ax13.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
              f'{val:.1f}', ha='center', va='bottom', fontsize=10)

ax13.set_title('需求波动统计', fontsize=12, fontweight='bold')
ax13.set_ylabel('数值')
ax13.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.suptitle('完整DROMARL：调整后的奖励函数与增强负载波动（集成博弈模型）', fontsize=16, fontweight='bold', y=1.02)
plt.show()

# ==================== 详细结果表格 ====================
print("\n" + "=" * 120)
print("最终调度详情 (调整后DROMARL策略)")
print("=" * 120)
print("时段 | 机组状态 | 发电(kW) | 需求(kW) | 偏差(kW) | 负载调整 | 电价概率 | 需求变化率")
print("-" * 120)

env.reset()
last_load = None
total_gen = 0
total_demand = 0
total_shortage = 0
total_surplus = 0
price_response_count = 0

for t in range(1, 13):
    a, _ = agent.select_action(env.get_state(), last_load, deterministic=True, evaluate=True)
    _, _, _, info = env.step(a[7], a[:7])
    last_load = torch.tensor([[info["load_ratio"]]], dtype=torch.float32)

    on_str = ''.join(['1' if x > 0.5 else '0' for x in info["actual_on"]])
    gap = info["gap"]
    demand_change_rate = ((info["demand"] - P_d_base[t - 1]) / P_d_base[t - 1] * 100)
    price_prob = info["price_prob"]

    # 检查价格响应
    if price_prob > 0.6 and info["load_ratio"] < -0.1:
        price_response_count += 1

    total_gen += info["gen"]
    total_demand += info["demand"]

    if gap < 0:
        total_shortage += abs(gap)
    elif gap > 0:
        total_surplus += gap

    print(
        f"{t:2d}   | {on_str} | {info['gen']:7.1f} | {info['demand']:8.1f} | {gap:7.1f} | {info['load_ratio']:+.3f} | {price_prob:7.3f} | {demand_change_rate:+.1f}%")

print("-" * 120)
print(f"总计 |         | {total_gen:7.1f} | {total_demand:8.1f} |         |")
print(f"平均 |         | {total_gen / 12:7.1f} | {total_demand / 12:8.1f} |         |")
print(f"缺电总量: {total_shortage:.1f} kW | 过剩总量: {total_surplus:.1f} kW")
print(f"总体平衡度: {(total_gen / total_demand * 100):.1f}%")
print(f"价格响应次数: {price_response_count}/12 次")

demand_variations = []
for t in range(1, 13):
    base = P_d_base[t - 1]
    actual = final_demand_curve[t - 1]
    variation = ((actual - base) / base * 100)
    demand_variations.append(abs(variation))

print(f"平均需求变化率: {np.mean(demand_variations):.1f}%")
print(f"最大需求变化率: {np.max(demand_variations):.1f}%")
print(f"需求标准差: {np.std(final_demand_curve):.2f} kW")
print(f"需求范围: {np.min(final_demand_curve):.1f} - {np.max(final_demand_curve):.1f} kW")

print("\n" + "=" * 120)
print("调整后DROMARL算法核心组件效果分析")
print("=" * 120)
print("1. 博弈模型集成:")
print(f"   - 供电侧均衡策略: {supply_strategy}")
print(f"   - 负载侧均衡策略: {load_strategy}")
print(f"   - 高电价概率分布: {DAY_AHEAD_HIGH_PRICE_PROB}")

print("\n2. 奖励函数调整:")
print(f"   - 缺电惩罚: -400*shortage - 20*shortage²")
print(f"   - 过剩惩罚: -2到-50*surplus")
print(f"   - 偏差惩罚: -30*abs_gap")
print(f"   - 奖励范围: {np.min(episode_rewards[-100:]):.1f} 到 {np.max(episode_rewards[-100:]):.1f}")
print(f"   - 平均奖励: {np.mean(episode_rewards[-100:]):.1f} (目标: 0到-3000)")

print("\n3. 负载波动增强:")
print(f"   - 最大调整幅度: ±{env.load_max_change * 100:.0f}%")
print(f"   - 平均调整幅度: {np.mean(np.abs(final_load_curve)):.3f}")
print(f"   - 调整变化率: {np.mean(load_changes):.3f}")
print(f"   - 惯性系数: {np.mean(agent.load_inertia_history[-100:]):.3f}")

print("\n4. 概率世界模型(PWM):")
print(f"   - 重构损失: {np.mean(agent.world_model_losses[-100:]):.4f}" if len(agent.world_model_losses) > 0 else "")

print("\n5. 分布鲁棒Critic:")
print(f"   - Critic损失: {np.mean(agent.critic_losses[-100:]):.4f}" if len(agent.critic_losses) > 0 else "")

print("\n6. 性能指标:")
print(f"   - 供需平衡率: {np.mean(final_balance):.1f}%")
print(f"   - 平均偏差: {np.mean(final_gaps):.2f} kW")
print(f"   - 舒适度违规率: {np.mean(final_comfort):.1f}%")
print(f"   - 价格响应率: {np.mean(response_effectiveness):.1%}")
print("=" * 120)

print("\n【系统总结】")
print("=" * 120)
print("已完成的功能整合：")
print("1. 博弈模型构建: 将充电站和电动汽车博弈修改为供电侧与负载侧博弈")
print("2. 参数修改: 拥堵系数→单位成本惩罚系数，单位运营成本→供电侧机组启停成本")
print("3. 时间扩展: 将三个时段扩展为十二个时间段")
print("4. 策略输出: 输出十二个时间段的供电侧策略p和负载侧策略q")
print("5. 系统融合: 博弈模型生成的高电价概率用于DROMARL调度系统")
print("6. 状态增强: 在VPP环境状态中添加博弈策略特征")
print("7. 需求响应增强: 负载侧响应考虑博弈策略影响")
print("=" * 120)