"""
主函数：调用上层调度和下层调度
运行整个电力市场博弈与VPP调度系统
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """主函数：运行整个系统"""

    print("=" * 120)
    print("电力市场博弈与VPP调度系统")
    print("=" * 120)

    # 检查是否已存在数据集
    dataset_file = 'electricity_market_dataset.csv'
    if os.path.exists(dataset_file):
        print(f"检测到已存在数据集文件: {dataset_file}")
        print("数据集信息:")
        df = pd.read_csv(dataset_file)
        print(f"  记录数: {len(df)}")
        print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"  特征数: {len(df.columns)}")

        # 显示数据集统计
        print(f"  基础电价概率均值: {df['base_price_prob'].mean():.4f}")
        print(f"  基础负载需求均值: {df['base_load_demand'].mean():.2f}MW")
        print(f"  温度范围: {df['temperature'].min():.1f}°C 到 {df['temperature'].max():.1f}°C")
        print(f"  天气分布: {dict(df['weather'].value_counts())}")
    else:
        print("未找到数据集文件，将创建新的数据集...")

    print("\n" + "=" * 120)
    print("第一阶段：上层调度 - 博弈模型训练")
    print("=" * 120)

    # 导入上层调度模块
    try:
        from game_theory_model import train_game_theory_model
    except ImportError as e:
        print(f"错误: 无法导入game_theory_model模块: {e}")
        print("请确保game_theory_model.py文件在同一目录下")
        return

    # 运行上层调度（博弈模型训练）
    print("开始训练博弈模型...")
    DAY_AHEAD_HIGH_PRICE_PROB, supply_strategy, load_strategy, SUPPLY_SCHEDULE, HOUR_DISTRIBUTION = train_game_theory_model()

    print("\n" + "=" * 120)
    print("博弈模型训练完成！")
    print("=" * 120)
    print(f"高电价概率分布 (前12小时): {DAY_AHEAD_HIGH_PRICE_PROB[:12]}")
    print(f"供电侧均衡策略: {supply_strategy}")
    print(f"负载侧均衡策略: {load_strategy}")
    print(f"供电时间分配: {HOUR_DISTRIBUTION}小时 (高/中/低)")

    # 保存博弈模型结果
    np.savez('game_theory_results.npz',
             day_ahead_high_price_prob=DAY_AHEAD_HIGH_PRICE_PROB,
             supply_strategy=supply_strategy,
             load_strategy=load_strategy,
             supply_schedule=SUPPLY_SCHEDULE,
             hour_distribution=HOUR_DISTRIBUTION)
    print("博弈模型结果已保存到: game_theory_results.npz")

    print("\n" + "=" * 120)
    print("第二阶段：上层调度 - VPP调度系统训练")
    print("=" * 120)

    # 导入下层调度模块
    try:
        from vpp_scheduling import train_vpp_scheduling
    except ImportError as e:
        print(f"错误: 无法导入vpp_scheduling模块: {e}")
        print("请确保vpp_scheduling.py文件在同一目录下")
        return

    # 运行下层调度（VPP调度训练）
    print("开始训练VPP调度系统...")
    agent, env, episode_rewards, balance_stats = train_vpp_scheduling(
        DAY_AHEAD_HIGH_PRICE_PROB,
        supply_strategy,
        load_strategy,
        SUPPLY_SCHEDULE,
        HOUR_DISTRIBUTION
    )

    print("\n" + "=" * 120)
    print("VPP调度系统训练完成！")
    print("=" * 120)

    # 计算最终性能
    final_window = min(100, len(episode_rewards))
    if len(episode_rewards) >= final_window:
        final_reward = np.mean(episode_rewards[-final_window:])
        final_balance = np.mean(balance_stats[-final_window:])
        print(f"最终平均奖励 (最近{final_window}轮): {final_reward:.1f}")
        print(f"最终平均平衡率 (最近{final_window}轮): {final_balance:.1f}%")

    print("\n" + "=" * 120)
    print("系统总结")
    print("=" * 120)
    print("已完成的功能整合：")
    print("1. 丰富的电力市场数据集创建 (electricity_market_dataset.csv)")
    print("2. 博弈模型构建: 供电侧与负载侧博弈均衡")
    print("3. 参数优化: 奖励函数系数减小，收敛值更合理")
    print("4. 时间扩展: 24小时时间段调度")
    print("5. 策略输出: 输出24小时电价概率和均衡策略")
    print("6. 系统融合: 博弈模型生成的高电价概率用于VPP调度")
    print("7. 状态增强: 在VPP环境状态中添加博弈策略特征")
    print("8. 需求响应增强: 负载侧响应考虑博弈策略和环境因素")

    print("\n生成的文件:")
    print("1. electricity_market_dataset.csv - 电力市场数据集")
    print("2. game_theory_results.npz - 博弈模型结果")
    print("3. game_theory_results.png - 博弈模型可视化")
    print("4. vpp_scheduling_results.png - VPP调度可视化")

    print("\n" + "=" * 120)
    print("系统运行完成！")
    print("=" * 120)


if __name__ == "__main__":
    main()