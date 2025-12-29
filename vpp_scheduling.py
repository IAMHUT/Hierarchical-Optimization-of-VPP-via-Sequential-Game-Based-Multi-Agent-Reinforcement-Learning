"""
下层调度：VPP调度系统训练
基于博弈模型的结果进行虚拟电厂调度优化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from collections import deque
import copy
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 从game_theory_model导入必要参数 ====================
try:
    from game_theory_model import (
        load_electricity_dataset,
        c_h, c_m, c_l,
        TIME_SLOTS
    )

    # 加载数据集获取基础负荷
    _, P_d_base_full, _, _, _ = load_electricity_dataset()
    P_d_base = list(P_d_base_full[:24])  # 取前24小时作为基础负荷
except ImportError:
    print("警告: 无法从game_theory_model导入，使用默认值")
    # 默认基础负荷
    P_d_base = [
        16.5, 14.2, 13.0, 12.5, 12.8, 14.5,  # 0-5时
        22.8, 30.5, 36.8, 40.5, 38.2, 34.8,  # 6-11时
        32.5, 29.8, 28.2, 29.5, 31.8, 33.5,  # 12-17时
        36.2, 42.5, 41.2, 35.8, 26.5, 19.2  # 18-23时
    ]
    TIME_SLOTS = 24

# ==================== VPP参数设定 ====================
# 发电机组参数 (基于实际机组数据)
gen_P = [15.0, 12.5, 8.5, 6.5, 4.0, 10.0, 12.0]  # 机组容量 (MW)
gen_cost = [75, 85, 100, 95, 110, 88, 82]  # 发电成本 (元/MWh)
N_GEN = 7
HORIZON = 24
MAX_EPISODES = 1500  # 进一步减少训练轮数
BUFFER_SIZE = 40000
BATCH_SIZE = 128
GAMMA = 0.99
LR_ACTOR = 2e-4
LR_CRITIC = 6e-4
LR_WORLD_MODEL = 8e-4
TAU = 0.005
UNCERTAINTY_RADIUS = 1.8  # 减小不确定性半径
PRICE_SENSITIVITY = 0.30  # 调整价格敏感性
DEMAND_VARIABILITY = 0.18
ROBUST_RADIUS = 0.03
GRADIENT_PENALTY_WEIGHT = 5.0  # 减小梯度惩罚权重
VARIANCE_PENALTY_WEIGHT = 0.0015  # 减小方差惩罚权重

STARTUP_COST = 130.0  # 减小启停成本
MIN_ON_TIME = [2, 3, 2, 4, 2, 3, 3]
MIN_OFF_TIME = [1, 2, 1, 3, 1, 2, 2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== VPP环境 ====================
class RealisticVPPEnv:
    def __init__(self, day_ahead_high_price_prob, supply_strategy, load_strategy,
                 supply_schedule, hour_distribution):
        self.gen_cost = gen_cost
        self.gen_P = gen_P
        self.min_on_time = MIN_ON_TIME
        self.min_off_time = MIN_OFF_TIME
        self.startup_cost = STARTUP_COST

        self.supply_strategy = supply_strategy
        self.load_strategy = load_strategy
        self.supply_schedule = supply_schedule
        self.hour_distribution = hour_distribution
        self.day_ahead_high_price_prob = day_ahead_high_price_prob

        self.state_dim = None
        self.reset()

    def reset(self):
        self.t = 0
        self.prev_on = np.zeros(N_GEN, dtype=np.float32)
        self.remain_min_on = np.array(self.min_on_time, dtype=np.float32)
        self.remain_min_off = np.zeros(N_GEN, dtype=np.float32)

        self.demand_history = []
        self.gen_history = []
        self.gap_history = []
        self.price_history = []
        self.load_ratio_history = []

        self.load_inertia = 0.55
        self.load_max_change = 0.35

        # 初始化负载偏好
        self.load_preference = np.zeros(HORIZON)
        for h in range(HORIZON):
            if h < len(self.supply_schedule[0]):
                if self.supply_schedule[0, h] > 0.5:
                    self.load_preference[h] = -0.25  # 高电价时偏好降低
                elif self.supply_schedule[2, h] > 0.5:
                    self.load_preference[h] = 0.15  # 低电价时偏好增加
                else:
                    self.load_preference[h] = 0.0

        self.elasticity = 0.25  # 减小价格弹性
        self.comfort_factor = 0.75
        self.weather_effect = np.random.normal(0, 0.05)  # 减小天气影响
        self.random_event_prob = 0.06

        return self.get_state()

    def get_state(self):
        if self.t >= HORIZON:
            if self.state_dim is not None:
                return np.zeros(self.state_dim, dtype=np.float32)
            else:
                pass

        onehot = np.zeros(HORIZON, dtype=np.float32)
        if self.t < HORIZON:
            onehot[self.t] = 1.0

        current_price_type = 1
        if self.t < self.supply_schedule.shape[1]:
            if self.supply_schedule[0, self.t] > 0.5:
                current_price_type = 0
            elif self.supply_schedule[1, self.t] > 0.5:
                current_price_type = 1
            else:
                current_price_type = 2

        if len(self.demand_history) > 0:
            recent_demand = self.demand_history[-min(5, len(self.demand_history)):]
            demand_avg = np.mean(recent_demand) if recent_demand else 0
            demand_std = np.std(recent_demand) if len(recent_demand) > 1 else 0
        else:
            demand_avg = 0
            demand_std = 0

        hour_feature = np.sin(2 * np.pi * self.t / 24)
        hour_feature2 = np.cos(2 * np.pi * self.t / 24)

        load_preference_feature = self.load_preference[self.t] if self.t < len(self.load_preference) else 0

        price_prob = self.day_ahead_high_price_prob[self.t] if self.t < len(self.day_ahead_high_price_prob) else 0.5

        state = np.concatenate([
            [self.t / HORIZON],
            self.prev_on,
            self.remain_min_on / 5.0,
            self.remain_min_off / 5.0,
            np.array(self.day_ahead_high_price_prob),
            onehot,
            [demand_avg / 50.0, demand_std / 20.0],
            [self.load_inertia, load_preference_feature],
            [price_prob, self.elasticity, self.comfort_factor],
            [hour_feature, hour_feature2, self.weather_effect],
            [self.load_max_change],
            self.supply_strategy,
            self.load_strategy,
            [current_price_type / 2.0]
        ])

        if self.state_dim is None:
            self.state_dim = len(state)

        return state.astype(np.float32)

    def calculate_realistic_demand(self, load_ratio_action):
        load_ratio = np.clip(load_ratio_action, -1.0, 1.0) * self.load_max_change

        if len(self.load_ratio_history) > 0:
            last_load_ratio = self.load_ratio_history[-1]
            load_ratio = self.load_inertia * last_load_ratio + (1 - self.load_inertia) * load_ratio

        base = P_d_base[self.t] if self.t < len(P_d_base) else P_d_base[-1]

        price_prob = self.day_ahead_high_price_prob[self.t] if self.t < len(self.day_ahead_high_price_prob) else 0.5

        price_response = -price_prob * self.elasticity * base * 0.8  # 减小价格响应

        if self.t < self.supply_schedule.shape[1]:
            if self.supply_schedule[0, self.t] > 0.5:
                price_response *= 1.2  # 减小高电价影响
            elif self.supply_schedule[2, self.t] > 0.5:
                price_response *= 0.8  # 减小低电价影响

        preference_effect = self.load_preference[self.t] * base if self.t < len(self.load_preference) else 0

        weather_effect = self.weather_effect * base * 0.4  # 减小天气影响

        uncertainty = np.random.normal(0, UNCERTAINTY_RADIUS * 0.6)  # 减小不确定性

        demand = (
                base * (1 + load_ratio)
                + price_response
                + preference_effect
                + weather_effect
                + uncertainty
        )

        demand = np.clip(demand, 8.0, 48.0)  # 调整范围

        comfort_penalty = 0
        if abs(load_ratio) > 0.25 * self.comfort_factor:  # 放宽舒适度限制
            comfort_penalty = -abs(load_ratio) * 20  # 减小舒适度惩罚

        return demand, load_ratio, comfort_penalty

    def step(self, load_ratio_raw, supply_raw):
        demand, actual_load_ratio, comfort_penalty = self.calculate_realistic_demand(load_ratio_raw)

        supply = supply_raw.copy()
        startup_cost = running_cost = 0.0

        current_price_type = 1
        if self.t < self.supply_schedule.shape[1]:
            if self.supply_schedule[0, self.t] > 0.5:
                current_price_type = 0
            elif self.supply_schedule[1, self.t] > 0.5:
                current_price_type = 1
            else:
                current_price_type = 2

        expected_supply_level = [1.0, 0.8, 0.6][current_price_type]

        # 机组调度逻辑
        for i in range(N_GEN):
            if self.remain_min_off[i] > 0:
                supply[i] = 0.0
                self.remain_min_off[i] = max(self.remain_min_off[i] - 1, 0)

            if supply[i] > 0.5 and self.prev_on[i] < 0.5:
                if self.remain_min_off[i] > 0:
                    supply[i] = 0.0
                else:
                    startup_cost += self.startup_cost * (0.9 + 0.1 * random.random())  # 减小启停成本波动

            if self.prev_on[i] > 0.5 and supply[i] < 0.5:
                if self.remain_min_on[i] > 0:
                    supply[i] = 1.0

            if supply[i] > 0.5:
                self.remain_min_on[i] = max(self.remain_min_on[i] - 1, 0)
                if self.prev_on[i] < 0.5:
                    self.remain_min_off[i] = 0
                running_cost += self.gen_cost[i] * 0.04 * (self.gen_P[i] / max(self.gen_P))  # 减小运行成本
            else:
                if self.prev_on[i] > 0.5:
                    self.remain_min_off[i] = self.min_off_time[i]
                self.remain_min_on[i] = self.min_on_time[i]

        # 计算总发电功率
        gen_power = np.sum(supply * np.array(self.gen_P))

        current_prev_on = self.prev_on.copy()
        self.prev_on = supply.copy()

        # 计算供需差距
        gap = gen_power - demand
        abs_gap = abs(gap)

        # 开关惩罚 - 减小惩罚系数
        switch_penalty = 0
        for i in range(N_GEN):
            if supply[i] > 0.5 and current_prev_on[i] < 0.5:
                switch_penalty += self.startup_cost * (0.6 + 0.2 * (self.gen_P[i] / max(self.gen_P)))  # 减小开关惩罚
            elif supply[i] < 0.5 and current_prev_on[i] > 0.5:
                switch_penalty += 40  # 减小关机惩罚

        # 惩罚分解 - 减小所有惩罚系数
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
            "comfort_penalty": comfort_penalty,
            "supply_schedule_penalty": 0
        }

        # 缺电惩罚 - 减小系数
        if gap < 0:
            shortage = -gap
            penalty_breakdown["shortage_penalty"] = -300 * shortage - 15 * shortage ** 2  # 减小缺电惩罚

        # 过剩惩罚 - 减小系数
        if gap > 0:
            surplus = gap
            if surplus <= 0.5:
                penalty_breakdown["surplus_penalty"] = -1.5 * surplus
            elif surplus <= 2.0:
                penalty_breakdown["surplus_penalty"] = -8 * surplus
            else:
                penalty_breakdown["surplus_penalty"] = -40 * surplus - 10 * surplus ** 2  # 减小过剩惩罚

        # 差距惩罚 - 减小系数
        penalty_breakdown["gap_penalty"] = -25 * abs_gap  # 减小差距惩罚

        # 浪费惩罚 - 减小系数
        if gap > 0:
            penalty_breakdown["waste_penalty"] = -60 * max(0, gap - 0.5)  # 减小浪费惩罚

        # 供电计划偏差惩罚 - 减小系数
        supply_level = np.mean(supply)
        schedule_deviation = abs(supply_level - expected_supply_level)
        penalty_breakdown["supply_schedule_penalty"] = -80 * schedule_deviation  # 减小计划偏差惩罚

        # 负载变化惩罚 - 减小系数
        if len(self.load_ratio_history) > 0:
            load_change = abs(actual_load_ratio - self.load_ratio_history[-1])
            penalty_breakdown["load_change_penalty"] = -3 * load_change ** 2  # 减小负载变化惩罚

        # 方差惩罚 - 减小系数
        action_variance = np.var(supply) + np.var([actual_load_ratio])
        penalty_breakdown["variance_penalty"] = -VARIANCE_PENALTY_WEIGHT * action_variance * 150  # 减小方差惩罚

        # 总奖励 - 应用衰减因子
        total_reward = sum(penalty_breakdown.values()) * 0.7  # 减小总奖励值

        # 记录历史
        self.demand_history.append(demand)
        self.gen_history.append(gen_power)
        self.gap_history.append(gap)
        self.price_history.append(
            self.day_ahead_high_price_prob[self.t] if self.t < len(self.day_ahead_high_price_prob) else 0)
        self.load_ratio_history.append(actual_load_ratio)

        self.t += 1
        done = self.t >= HORIZON

        next_state = self.get_state()

        return next_state, float(total_reward), done, {
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
            "price_prob": self.day_ahead_high_price_prob[self.t - 1] if self.t > 0 else 0,
            "expected_supply_level": expected_supply_level,
            "actual_supply_level": supply_level
        }


# ==================== 世界模型 ====================
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

        eps = torch.randn_like(enc_mean) * 0.08  # 减小噪声
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
        reward_loss = F.mse_loss(reward_pred, reward_target) * 0.8  # 减小奖励损失权重

        total_loss = recon_loss + 0.08 * kl_div + 0.08 * reward_loss  # 减小KL和奖励损失权重

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'kl_div': kl_div.item(),
            'reward_loss': reward_loss.item()
        }


# ==================== 改进的负载Actor ====================
class InertialLoadActor(nn.Module):
    def __init__(self, s_dim, noise_scale=0.15, dropout_prob=0.1):  # 减小噪声
        super().__init__()
        self.s_dim = s_dim
        self.noise_scale = noise_scale
        self.dropout_prob = dropout_prob

        self.feature_extractor = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.inertia_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.action_net = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        self.uncertainty_net = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x, last_action=None, deterministic=False, training=True):
        if x.dtype != torch.float32:
            x = x.float()

        if training and self.training:
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise

        features = self.feature_extractor(x)
        inertia = self.inertia_net(features)

        if last_action is not None:
            if last_action.device != x.device:
                last_action = last_action.to(x.device)
            if last_action.dtype != torch.float32:
                last_action = last_action.float()

            inertia_features = torch.cat([features, inertia], dim=1)
            raw_action = self.action_net(inertia_features)

            action = inertia * last_action + (1 - inertia) * raw_action
        else:
            inertia_features = torch.cat([features, inertia], dim=1)
            action = self.action_net(inertia_features)

        uncertainty = self.uncertainty_net(features)

        if deterministic or not training:
            return torch.clamp(action, -1.0, 1.0), uncertainty, inertia
        else:
            exploration_noise = torch.randn_like(action) * 0.12  # 减小探索噪声
            action = torch.clamp(action + exploration_noise, -1.0, 1.0)
            return action, uncertainty, inertia


# ==================== 不确定性感知供应Actor ====================
class UncertaintyAwareSupplyActor(nn.Module):
    def __init__(self, s_dim, noise_scale=0.15, dropout_prob=0.1):  # 减小噪声
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
        if state.dtype != torch.float32:
            state = state.float()
        if action.dtype != torch.float32:
            action = action.float()

        x = torch.cat([state, action], dim=1)
        return self.net(x)

    def compute_gradient_penalty(self, real_samples, generated_samples, state):
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


# ==================== DROMARL算法 ====================
class DROMARL:
    def __init__(self, s_dim):
        self.s_dim = s_dim

        self.load_actor = InertialLoadActor(s_dim).to(device)
        self.supply_actor = UncertaintyAwareSupplyActor(s_dim).to(device)
        self.critic = DistributionRobustCritic(s_dim).to(device)
        self.world_model = ProbabilisticWorldModel(s_dim, N_GEN + 1, h_dim=128, z_dim=32).to(device)

        self.load_actor_target = copy.deepcopy(self.load_actor)
        self.supply_actor_target = copy.deepcopy(self.supply_actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.opt_load = optim.Adam(self.load_actor.parameters(), lr=LR_ACTOR)
        self.opt_supply = optim.Adam(self.supply_actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.opt_world_model = optim.Adam(self.world_model.parameters(), lr=LR_WORLD_MODEL)

        self.buffer = deque(maxlen=BUFFER_SIZE)

        self.temperature = 0.8  # 降低初始温度
        self.noise_scale = 0.2  # 降低初始噪声
        self.uncertainty_threshold = 0.75  # 降低不确定性阈值

        self.critic_losses = []
        self.actor_losses = []
        self.world_model_losses = []
        self.uncertainty_history = []
        self.load_inertia_history = []

    def select_action(self, state, last_load_action=None, deterministic=False, evaluate=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)

        if last_load_action is not None:
            if isinstance(last_load_action, torch.Tensor):
                last_load_action = last_load_action.to(device).float()
            else:
                last_load_action = torch.FloatTensor(last_load_action).unsqueeze(0).to(device)

        with torch.no_grad():
            # 负载侧动作
            if evaluate:
                self.load_actor.eval()
                load_action, _, inertia = self.load_actor(s, last_load_action, deterministic=True, training=False)
                self.load_actor.train()
            else:
                load_action, _, inertia = self.load_actor(s, last_load_action, deterministic=False, training=True)

            load_action = load_action.float()

            # 供应侧动作
            s_with_load = torch.cat([s, load_action], dim=1).float()

            if evaluate or deterministic:
                self.supply_actor.eval()
                supply_logits, _ = self.supply_actor(s_with_load, training=False)
                self.supply_actor.train()
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

        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 0.8)  # 减小梯度裁剪
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
            last_load_actions = a[:, -1].unsqueeze(1).to(device)

            self.load_actor_target.eval()
            next_load, _, _ = self.load_actor_target(s_next, last_load_actions, deterministic=True, training=False)
            self.load_actor_target.train()
            next_load = next_load.float()

            s_next_aug = torch.cat([s_next, next_load], dim=1).float()
            self.supply_actor_target.eval()
            next_logits, _ = self.supply_actor_target(s_next_aug, training=False)
            self.supply_actor_target.train()
            next_supply_prob = F.softmax(next_logits / 0.08, dim=-1)[:, :, 1]  # 降低温度
            next_action = torch.cat([next_supply_prob, next_load], dim=1)

            target_q = self.critic_target(s_next, next_action)
            y = r + GAMMA * (1 - done) * target_q

        # 分布鲁棒Critic训练
        with torch.no_grad():
            noise = torch.randn_like(a) * ROBUST_RADIUS
            a_perturbed = torch.clamp(a + noise, -1.0, 1.0)

        gradient_penalty = self.critic.compute_gradient_penalty(a, a_perturbed, s)

        critic_loss = F.mse_loss(current_q, y) * 0.9 + GRADIENT_PENALTY_WEIGHT * gradient_penalty  # 减小Critic损失

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.8)  # 减小梯度裁剪
        self.opt_critic.step()

        # 更新Actor
        # 负载Actor更新
        load_action, load_uncertainty, inertia = self.load_actor(s, None, deterministic=False, training=True)
        load_action = load_action.float()
        s_aug = torch.cat([s, load_action], dim=1).float()
        supply_logits, supply_uncertainty = self.supply_actor(s_aug, training=True)
        supply_prob = F.gumbel_softmax(supply_logits, tau=self.temperature, hard=False)[:, :, 1]
        predicted_action = torch.cat([supply_prob, load_action], dim=1)

        actor_loss_load = -self.critic(s, predicted_action).mean() * 0.9  # 减小Actor损失

        if load_uncertainty.mean() > self.uncertainty_threshold:
            actor_loss_load += 0.015 * load_uncertainty.mean()  # 减小不确定性惩罚

        self.opt_load.zero_grad()
        actor_loss_load.backward()
        torch.nn.utils.clip_grad_norm_(self.load_actor.parameters(), 0.5)
        self.opt_load.step()

        # 供应Actor更新
        with torch.no_grad():
            self.load_actor.eval()
            best_load, _, _ = self.load_actor(s, None, deterministic=True, training=False)
            self.load_actor.train()
            best_load = best_load.float()

        s_best = torch.cat([s, best_load], dim=1).float()
        supply_logits2, supply_uncertainty2 = self.supply_actor(s_best, training=True)
        supply_prob2 = F.gumbel_softmax(supply_logits2, tau=self.temperature, hard=False)[:, :, 1]
        final_action = torch.cat([supply_prob2, best_load], dim=1)

        actor_loss_supply = -self.critic(s, final_action).mean() * 0.9  # 减小Actor损失

        action_variance = torch.var(final_action, dim=1).mean()
        actor_loss_supply += VARIANCE_PENALTY_WEIGHT * action_variance * 0.8  # 减小方差惩罚

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
        self.temperature = max(0.25, self.temperature * 0.9995)  # 更慢的衰减
        self.noise_scale = max(0.06, self.noise_scale * 0.9998)  # 更慢的衰减

        # 记录
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append((actor_loss_load.item() + actor_loss_supply.item()) / 2)
        self.load_inertia_history.append(inertia.mean().item())

        return critic_loss.item(), (actor_loss_load.item() + actor_loss_supply.item()) / 2, 0.0


# ==================== VPP调度主训练函数 ====================
def train_vpp_scheduling(day_ahead_high_price_prob, supply_strategy, load_strategy,
                         supply_schedule, hour_distribution):
    """训练VPP调度系统"""

    print(f"使用设备: {device}")

    # 先创建环境来获取状态维度
    env = RealisticVPPEnv(day_ahead_high_price_prob, supply_strategy, load_strategy,
                          supply_schedule, hour_distribution)
    s_dim = len(env.get_state())
    print(f"状态维度: {s_dim}")
    agent = DROMARL(s_dim)

    print("\n" + "=" * 120)
    print("开始VPP调度系统训练...")
    print("基于具体电力市场数据集的改进版本")
    print("奖励函数已减小，使用具体数据集")
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

            if len(a) >= 8:
                ns, r, done, info = env.step(a[7], a[:7])
            else:
                default_load_ratio = 0.0
                default_supply = np.zeros(N_GEN)
                ns, r, done, info = env.step(default_load_ratio, default_supply)

            agent.store(s, a, r, ns, float(done))

            world_model_loss, wm_loss_details = agent.update_world_model()

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

        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_balance = np.mean(balance_stats[-50:]) if len(balance_stats) >= 50 else np.mean(balance_stats)
            avg_gap = np.mean(episode_gaps[-50:]) if len(episode_gaps) >= 50 else np.mean(episode_gaps)
            avg_load_adj = np.mean(load_adjustments[-50:]) if len(load_adjustments) >= 50 else np.mean(load_adjustments)
            avg_comfort_violation = np.mean(comfort_violations[-50:]) if len(comfort_violations) >= 50 else np.mean(
                comfort_violations)
            print(f"Episode {ep:4d} | 奖励: {avg_reward:+.1f} | 平衡率: {avg_balance:.1f}% | "
                  f"偏差: {avg_gap:.2f}kW | 负载调整: {avg_load_adj:.3f} | 舒适违规: {avg_comfort_violation:.1f}%")

    # ==================== 性能分析 ====================
    print("\n" + "=" * 120)
    print("训练完成！性能分析报告")
    print("=" * 120)

    window = min(50, len(episode_rewards))
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
    print(f"  - 平均奖励: {np.mean(episode_rewards[-window:]):.1f} (已优化到合理范围)")

    # ==================== 可视化结果 ====================
    fig = plt.figure(figsize=(20, 16))

    # 1. 奖励收敛曲线
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(episode_rewards, alpha=0.6, color='lightblue', label='单轮奖励')
    if len(episode_rewards) > 50:
        smoothed = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
        ax1.plot(range(49, len(episode_rewards)), smoothed, color='red', linewidth=2, label='滑动平均')
    ax1.set_title('奖励收敛曲线 (奖励值已减小)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('累计奖励')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 添加奖励范围线
    reward_mean = np.mean(episode_rewards[-window:]) if len(episode_rewards) >= window else np.mean(episode_rewards)
    ax1.axhline(y=reward_mean, color='green', linestyle='--', alpha=0.5, label=f'平均奖励: {reward_mean:.1f}')

    # 2. 供需平衡率趋势
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(balance_stats, alpha=0.6, color='green')
    if len(balance_stats) > 50:
        smoothed_balance = np.convolve(balance_stats, np.ones(50) / 50, mode='valid')
        ax2.plot(range(49, len(balance_stats)), smoothed_balance, color='darkgreen', linewidth=2)
    ax2.set_title('供需平衡率趋势', fontsize=12, fontweight='bold')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('平衡率 (%)')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 105)

    # 3. 负载调整幅度趋势
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(load_adjustments, alpha=0.6, color='purple')
    if len(load_adjustments) > 50:
        smoothed_load = np.convolve(load_adjustments, np.ones(50) / 50, mode='valid')
        ax3.plot(range(49, len(load_adjustments)), smoothed_load, color='darkviolet', linewidth=2)
    ax3.set_title('负载调整幅度趋势', fontsize=12, fontweight='bold')
    ax3.set_xlabel('训练轮数')
    ax3.set_ylabel('负载调整幅度')
    ax3.grid(alpha=0.3)

    # 4. 最终调度结果
    ax4 = plt.subplot(3, 3, (4, 6))
    final_demand_curve = []
    final_gen_curve = []
    final_load_curve = []

    env.reset()
    last_load = None
    for t in range(24):
        a, _ = agent.select_action(env.get_state(), last_load, deterministic=True, evaluate=True)

        if len(a) >= 8:
            _, _, _, info = env.step(a[7], a[:7])
        else:
            default_load_ratio = 0.0
            default_supply = np.zeros(N_GEN)
            _, _, _, info = env.step(default_load_ratio, default_supply)

        final_demand_curve.append(info["demand"])
        final_gen_curve.append(info["gen"])
        final_load_curve.append(info["load_ratio"])
        last_load = torch.tensor([[info["load_ratio"]]], dtype=torch.float32)

    hours = range(1, 25)
    ax4.plot(hours, final_demand_curve, 'o-', color='blue', linewidth=3, markersize=8, label='实际需求')
    ax4.plot(hours, final_gen_curve, 's-', color='red', linewidth=3, markersize=8, label='发电功率')

    ax4.fill_between(hours, final_demand_curve, final_gen_curve,
                     where=np.array(final_gen_curve) >= np.array(final_demand_curve),
                     alpha=0.3, color='green', label='发电过剩')
    ax4.fill_between(hours, final_demand_curve, final_gen_curve,
                     where=np.array(final_gen_curve) < np.array(final_demand_curve),
                     alpha=0.3, color='red', label='供电不足')

    ax4.set_title('最终调度结果：发电与需求匹配 (基于电力市场数据集)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('时间段 (小时)')
    ax4.set_ylabel('功率 (kW)')
    ax4.grid(alpha=0.3)
    ax4.legend()
    ax4.set_xticks(range(1, 25, 2))

    # 添加基础负载线
    ax4.plot(hours, P_d_base[:24], '--', color='gray', linewidth=1.5, alpha=0.7, label='基础负载需求')

    # 5. 负载调整与电价关系
    ax5 = plt.subplot(3, 3, 7)
    width = 0.35
    x = np.arange(24)
    ax5.bar(x - width / 2, day_ahead_high_price_prob[:24], width, color='orange', alpha=0.7, label='高电价概率')
    ax5.bar(x + width / 2, final_load_curve, width, color='blue', alpha=0.7, label='负载调整率')
    ax5.set_title('电价概率 vs 负载调整', fontsize=12, fontweight='bold')
    ax5.set_xlabel('时间段')
    ax5.set_ylabel('概率 / 调整率')
    ax5.grid(alpha=0.3, axis='y')
    ax5.legend()
    ax5.set_xticks(range(0, 24, 3))

    # 6. 博弈策略可视化
    ax6 = plt.subplot(3, 3, 8)
    x_pos = np.arange(3)
    width = 0.35

    ax6.bar(x_pos - width / 2, supply_strategy, width, color='red', alpha=0.7, label='供电侧策略')
    ax6.bar(x_pos + width / 2, load_strategy, width, color='blue', alpha=0.7, label='负载侧策略')

    ax6.set_xlabel('电价类型')
    ax6.set_ylabel('概率')
    ax6.set_title('博弈均衡策略对比', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(['高电价', '中电价', '低电价'])
    ax6.legend()
    ax6.grid(alpha=0.3, axis='y')

    # 7. 需求变化分析
    ax7 = plt.subplot(3, 3, 9)
    if len(P_d_base) >= 24:
        base_vs_actual = [(final_demand_curve[i] - P_d_base[i]) / P_d_base[i] * 100
                          for i in range(24)]
        hours_plot = range(1, 25)
        ax7.plot(hours_plot, base_vs_actual, 'o-', color='green', linewidth=2, markersize=6)
        ax7.fill_between(hours_plot, 0, base_vs_actual, where=np.array(base_vs_actual) > 0,
                         alpha=0.3, color='red', label='需求增加')
        ax7.fill_between(hours_plot, 0, base_vs_actual, where=np.array(base_vs_actual) < 0,
                         alpha=0.3, color='blue', label='需求减少')
        ax7.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax7.set_title('需求相对变化率 (%)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('时间段')
        ax7.set_ylabel('变化率 (%)')
        ax7.grid(alpha=0.3)
        ax7.legend()
    else:
        ax7.text(0.5, 0.5, '基础负荷数据不足', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('需求相对变化率', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.suptitle('完整DROMARL调度系统（基于具体电力市场数据集）', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('vpp_scheduling_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==================== 详细结果表格 ====================
    print("\n" + "=" * 120)
    print("最终调度详情")
    print("=" * 120)
    print("时段 | 机组状态 | 发电(kW) | 需求(kW) | 偏差(kW) | 负载调整 | 电价概率 | 需求变化率")
    print("-" * 120)

    env.reset()
    last_load = None
    total_gen = 0
    total_demand = 0
    total_shortage = 0
    total_surplus = 0

    for t in range(1, 25):
        a, _ = agent.select_action(env.get_state(), last_load, deterministic=True, evaluate=True)

        if len(a) >= 8:
            _, _, _, info = env.step(a[7], a[:7])
        else:
            default_load_ratio = 0.0
            default_supply = np.zeros(N_GEN)
            _, _, _, info = env.step(default_load_ratio, default_supply)

        last_load = torch.tensor([[info["load_ratio"]]], dtype=torch.float32)

        on_str = ''.join(['1' if x > 0.5 else '0' for x in info["actual_on"]])
        gap = info["gap"]
        demand_change_rate = ((info["demand"] - P_d_base[t - 1]) / P_d_base[t - 1] * 100) if t <= len(P_d_base) else 0
        price_prob = info["price_prob"]

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
    print(f"平均 |         | {total_gen / 24:7.1f} | {total_demand / 24:8.1f} |         |")
    print(f"缺电总量: {total_shortage:.1f} kW | 过剩总量: {total_surplus:.1f} kW")
    print(f"总体平衡度: {(total_gen / total_demand * 100):.1f}%")

    return agent, env, episode_rewards, balance_stats