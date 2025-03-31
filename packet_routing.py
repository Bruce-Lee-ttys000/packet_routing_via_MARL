from xuance.environment import RawMultiAgentEnv
from collections import deque
import numpy as np
import random
from gymnasium.spaces import Box, Discrete
import networkx as nx
import os
import csv
from xuance.environment.multi_agent_env.satellite_network import create_satellite_network  # 导入外部函数
import time


class Packet:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.hops = 0
        self.nodes = [source]
        self.actions = []
        self.rewards = []
        self.visited_nodes = set()  # 已访问节点的集合
        self.visited_nodes.add(source)  # 初始化时将源节点加入集合
        self.arrival_time = None  # 到达目的节点的时间
        self.transmission_delays = []  # 传输时延记录
        self.propagation_delays = []  # 传播时延记录
        self.queue_delays = []  # 排队时延记录
        self.enqueue_time = None  # 进入队列的时间步
        self.queue_delay_steps = 0  # 累计排队的步数
        self.packet_id = None
        self.path_delays = []
        self.total_delay = 0
        self.start_time = time.time()
        self.max_hops = 0  # 最大跳数
        self.is_dropped = False  # 丢包标志
        self.drop_reason = None  # 丢包原因

    def add_hop(self, next_node, delay):
        self.nodes.append(next_node)
        self.visited_nodes.add(next_node)
        self.path_delays.append(delay)
        self.total_delay += delay
        self.hops += 1

    def print_path_info(self):
        print(f"\n数据包 {self.packet_id} 的传输信息:")
        print(f"转发路径: {' -> '.join(map(str, self.nodes))}")
        print(f"跳数: {self.hops}")
        print(f"每跳延迟 (ms): {[f'{delay:.2f}' for delay in self.path_delays]}")
        print(f"总延迟 (ms): {self.total_delay:.2f}")
        print(f"已访问节点: {sorted(list(self.visited_nodes))}")
        if self.arrival_time:
            transmission_time = self.arrival_time - self.start_time
            print(f"传输总时间: {transmission_time:.2f}秒")


class PacketRoutingEnv(RawMultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

        # 从 config 中提取参数
        self.max_episode_steps = getattr(config, 'max_steps', 150)
        self.seed = getattr(config, 'seed', 64)
        self.common_reward = getattr(config, 'common_reward', False)
        self.reward_scalarisation = getattr(config, 'reward_scalarisation', "mean")
        self.algorithm = getattr(config, 'algorithm', 'mappo')
        self.enable_topology_clipping = getattr(config, 'enable_topology_clipping', True)

        # 奖励和惩罚参数配置
        self.rewards_config = {
            # ✅ 1. 维持基础奖励
            'step_penalty': -1,

            # ✅ 2. 提高奖励，让智能体更积极寻求全局最优
            'packet_arrival_reward': 100,  # 提高单个数据包到达奖励
            'path_completion_reward': 2000,

            # ✅ 3. 强化路径优化策略
            'distance_reward_weight': 50,  # 增加向目标移动的奖励权重
            'distance_penalty_weight': -25,  # 相应调整惩罚权重

            # ✅ 4. 提高错误决策惩罚
            'loop_penalty': -30,  # 增加循环惩罚
            'queue_full_penalty': -15,
            'drop_penalty': -10,
            'wrong_destination_penalty': -20,
            'max_hops_penalty': -20,

            # ✅ 5. 到达率里程碑奖励（指数增长）
            'arrival_20_reward': 200,     # 20%数据包到达
            'arrival_40_reward': 400,     # 40%数据包到达
            'arrival_60_reward': 800,     # 60%数据包到达
            'arrival_80_reward': 1600,    # 80%数据包到达
            'arrival_100_reward': 3200,   # 100%数据包到达
        }

        # 到达率里程碑设置
        self.arrival_milestones = {
            0.20: 'arrival_20_reward',   # 20%
            0.40: 'arrival_40_reward',   # 40%
            0.60: 'arrival_60_reward',   # 60%
            0.80: 'arrival_80_reward',   # 80%
            1.00: 'arrival_100_reward',  # 100%
        }
        
        # 添加到达率里程碑追踪
        self.reached_arrival_milestones = set()

        # 卫星网络拓扑参数
        self.number_of_orbital_planes = getattr(config, 'number_of_orbital_planes', 4)
        self.number_of_satellites_per_plane = getattr(config, 'number_of_satellites_per_plane', 43)
        self.satellite_height = getattr(config, 'satellite_height', 560)

        # 全局计数器
        self.global_episode_count = 0
        self.episode_done = False

        # 带宽设置
        self.bandwidth = getattr(config, 'bandwidth', 10e9)
        self.packet_size = 1500 * 8
        self.bandwidth_in_packets_per_sec = self.bandwidth / self.packet_size

        # 图形和网络初始化
        self.graph, self.source_node, self.destination_node = self.create_graph()
        self.current_step = 0

        # 计算最短路径和最大跳数
        shortest_path = nx.shortest_path(self.graph, self.source_node, self.destination_node)
        self.shortest_path_length = len(shortest_path) - 1  # 减1是因为路径长度包含了起始节点
        self.max_hops = self.shortest_path_length * 2  # 设置最大跳数为最短路径长度的2倍
        print(f"最大允许跳数: {self.max_hops}")

        # 数据包参数
        self.packets = getattr(config, 'packets', 50)
        self.received_packets_count = 0
        self.packet_sequence_number = 1

        # 队列容量和结构
        self.queue_capacity = getattr(config, 'queue_capacity', 10)
        self.queues = {
            node: deque(maxlen=self.packets) if node in [self.source_node, self.destination_node] else deque(
                maxlen=self.queue_capacity)
            for node in self.graph.nodes()
        }

        # 计算每个节点到目的节点的最短路径长度
        self.shortest_paths = nx.single_source_shortest_path_length(self.graph, self.destination_node)
        self.neighbours = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}

        self.agents = [f"agent_{node}" for node in sorted(self.graph.nodes())]
        self.possible_agents = self.agents[:]
        self.num_agents = len(self.agents)
        self.agent_groups = [self.agents]

        max_neighbors = max(len(neighbors) for neighbors in self.neighbours.values())
        obs_space_size = 1 + max_neighbors * 3 + 1
        self.observation_space = {
            agent: Box(
                low=np.zeros(obs_space_size, dtype=np.float32),
                high=np.ones(obs_space_size, dtype=np.float32) * np.float32(np.inf),
                shape=(obs_space_size,),
                dtype=np.float32
            ) for agent in self.agents
        }

        # 修改动作空间为一维，只需要选择邻居节点
        self.action_space = {
            agent: Box(
                low=np.zeros(1, dtype=np.float32),
                high=np.ones(1, dtype=np.float32),
                shape=(1,),
                dtype=np.float32
            ) for agent in self.agents
        }

        self.state_space = Box(
            low=np.zeros(obs_space_size * len(self.agents), dtype=np.float32),  # 明确指定 dtype
            high=np.ones(obs_space_size * len(self.agents), dtype=np.float32) * np.float32(np.inf),  # 明确指定 dtype
            shape=(obs_space_size * len(self.agents),),
            dtype=np.float32
        )

        # 奖励参数初始化
        self.best_total_delay = float('inf')
        self.best_reward = float('-inf')

        # 奖励参数调整
        self.distance_reward_weight = getattr(config, 'distance_reward_weight', 100)  # 增加向目标移动的奖励
        self.distance_penalty_weight = getattr(config, 'distance_penalty_weight', -100)  # 加大远离目标的惩罚
        self.loop_penalty = getattr(config, 'loop_penalty', -50)  # 加大循环路径的惩罚
        self.packet_arrival_reward = getattr(config, 'packet_arrival_reward', 200)  # 单个数据包到达的奖励
        self.path_completion_reward = getattr(config, 'path_completion_reward', 1000)  # 所有数据包都到达的奖励

        # 统计相关参数
        self.episode_count = 0
        self.final_rewards_history = []
        self.last_packet_arrival_time = None

        # 修改文件路径
        base_path = os.path.dirname(__file__)
        self.csv_file = os.path.join(base_path, f'results_{self.algorithm}.csv')
        self.plot_file = os.path.join(base_path, f'rewards_plot_{self.algorithm}.html')

        # 初始化CSV文件
        self.initialize_csv()

        # 重置环境
        self.episode_total_reward = 0.0  # 当前回合的总奖励
        self.episode_agent_rewards = {agent: 0.0 for agent in self.agents}  # 每个智能体的回合奖励

        # 添加数据包跟踪
        self.all_packets = {}  # 用于跟踪所有数据包
        self.current_episode_packets = set()  # 当前回合的数据包ID集合

        # 添加计数器
        self.dropped_packets_count = 0  # 因超过跳数限制而丢弃的数据包数量

        self.reset()

    def reset(self, **kwargs):
        # 重置数据包跟踪
        self.all_packets = {}
        self.current_episode_packets = set()
        self.packet_sequence_number = 1

        # 更新全局回合和步数计数器
        self.current_step = 0
        self.received_packets_count = 0
        self.episode_done = False

        # 初始化个体回合奖励字典
        self.individual_episode_reward = {agent: 0.0 for agent in self.agents}

        # 设置随机种子
        seed = kwargs.get("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 初始化奖励、终止标志、截断标志和额外信息
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # 初始化队列状态
        self.queues = {
            node: deque(maxlen=self.packets) if node in [self.source_node, self.destination_node] else deque(
                maxlen=self.queue_capacity)
            for node in self.graph.nodes()
        }

        # 初始化通道状态
        self.channels = {
            node: {neighbour: deque() for neighbour in self.neighbours[node]}
            for node in self.graph.nodes()
        }

        # 在源节点生成数据包
        self.generate_packets(num_packets=self.packets)

        # 创建每个代理的初始观测值
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}

        # 重置回合总奖励
        self.episode_total_reward = 0.0
        self.episode_agent_rewards = {agent: 0.0 for agent in self.agents}

        # 重置计数器
        self.dropped_packets_count = 0

        # 重置到达率里程碑
        self.reached_arrival_milestones = set()

        # 返回初始观测值和环境信息
        return self.observations, self.infos

    def step(self, action_dict):
        """执行环境的一步"""
        self.current_step += 1
        observations = {}
        rewards = {agent: 0 for agent in self.agents}
        terminations = {}
        infos = {agent: {} for agent in self.agents}

        # print(f"\n=== Step {self.current_step} ===")

        # **执行所有智能体的动作**
        for agent_id, action in action_dict.items():
            if self.terminations[agent_id]:
                continue
            node = int(agent_id.split('_')[1])
            packet_reached_destination = self.perform_action(node, action)
            rewards[agent_id] += self.rewards[agent_id]
            # 累加每个智能体的回合奖励
            self.episode_agent_rewards[agent_id] += rewards[agent_id]

        # **执行数据包转移**
        self.transfer_packets()

        # **检查队列状态**
        queue_lengths = {node: len(self.queues[node]) for node in self.graph.nodes() if len(self.queues[node]) > 0}

        # **检查到达率里程碑**
        self.check_arrival_milestones()

        # **检查是否所有数据包都已处理完毕（到达目的地或被丢弃）**
        total_processed_packets = self.received_packets_count + self.dropped_packets_count
        if total_processed_packets == self.packets:
            success_rate = (self.received_packets_count / self.packets) * 100
            print(f"\n✅ 所有数据包处理完毕！")
            print(f"成功到达: {self.received_packets_count} ({success_rate:.1f}%)")
            print(f"因跳数限制丢弃: {self.dropped_packets_count} ({100-success_rate:.1f}%)")
            print(f"总共用了 {self.current_step} 步")
            
            for agent_id in self.agents:
                self.terminations[agent_id] = True

        # **更新观测**
        for agent_id in self.agents:
            observations[agent_id] = self.observe(agent_id)
            terminations[agent_id] = self.terminations[agent_id]

        # **累加当前步骤所有智能体的奖励**
        step_total_reward = sum(rewards.values())
        self.episode_total_reward += step_total_reward

        # **达到最大步数时终止**
        truncated = self.current_step >= self.max_episode_steps

        # 修改回合结束的条件：所有智能体都终止或达到最大步数或所有数据包都已处理完毕
        episode_done = all(terminations.values()) or truncated or total_processed_packets == self.packets

        if episode_done:
            self.global_episode_count += 1
            self.log_rewards_to_csv(self.global_episode_count, self.episode_total_reward)
            self.print_episode_summary()

        return observations, rewards, terminations, truncated, infos

    def state(self):
        """返回环境的全局状态"""
        try:
            # 获取所有智能体的观测值并拼接
            state = np.concatenate([self.observe(agent) for agent in self.agents])
            return state
        except Exception as e:
            raise RuntimeError(f"Error in getting state: {e}")

    def agent_mask(self):
        """返回智能体的有效性掩码"""
        return {agent: not self.terminations[agent] for agent in self.agents}

    def avail_actions(self):
        """返回每个智能体可用动作的掩码"""
        avail_actions = {}
        for agent in self.agents:
            node = int(agent.split('_')[1])
            num_neighbors = len(self.neighbours[node])
            # 对于连续动作空间，返回True
            avail_actions[agent] = np.ones(self.action_space[agent].shape, dtype=np.bool_)
        return avail_actions

    def render(self, mode="human"):
        """渲染环境"""
        pass

    def close(self):
        """关闭环境"""
        pass

    def initialize_csv(self):
        """初始化 CSV 文件，写入表头"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Total Reward"])

    def log_rewards_to_csv(self, episode, total_reward):
        """将每回合的奖励总和记录到 CSV 文件"""
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, total_reward])

    def create_graph(self):
        return create_satellite_network(self)

    def generate_packets(self, num_packets=None):
        """生成数据包"""
        if num_packets is None:
            num_packets = self.packets

        for i in range(num_packets):
            packet = Packet(self.source_node, self.destination_node)  # 使用self.destination_node作为目的节点
            packet.packet_id = f"P{self.packet_sequence_number}"
            # 使用类级别的最大跳数
            packet.max_hops = self.max_hops
            self.packet_sequence_number += 1
            self.queues[self.source_node].append(packet)
            self.all_packets[packet.packet_id] = packet
            self.current_episode_packets.add(packet.packet_id)

    def transfer_packets(self):
        """从通道中移动数据包到目标节点队列"""
        # 创建临时字典来存储需要处理的数据包
        pending_packets = {}

        # 第一步：收集所有需要处理的数据包
        for node in self.channels:
            for neighbour, packets in list(self.channels[node].items()):
                while packets:
                    packet = packets.popleft()
                    if neighbour not in pending_packets:
                        pending_packets[neighbour] = []
                    pending_packets[neighbour].append((node, packet))

        # 第二步：处理每个节点的数据包
        for target_node, packet_list in pending_packets.items():
            # 处理数据包
            for source_node, packet in packet_list:
                # 实时计算目标节点的剩余容量
                current_queue_length = len(self.queues[target_node])
                if target_node in [self.source_node, self.destination_node]:
                    remaining_capacity = self.packets - current_queue_length
                else:
                    remaining_capacity = self.queue_capacity - current_queue_length

                if remaining_capacity > 0:
                    self.queues[target_node].append(packet)
                else:
                    # 目标队列满时，将数据包放回原节点
                    if source_node != self.source_node:  # 如果不是源节点，才放回原节点
                        self.queues[source_node].append(packet)
                        # 对所有参与转发的节点施加较小的惩罚
                        for visited_node in packet.visited_nodes:
                            self.rewards[f"agent_{visited_node}"] += self.rewards_config[
                                                                         'queue_full_penalty'] / 2  # 减小惩罚力度

        # 清空所有通道
        for node in self.channels:
            for neighbour in self.channels[node]:
                self.channels[node][neighbour].clear()

    def observe(self, agent_id):
        node = int(agent_id.split('_')[1])
        obs_size = self.observation_space[agent_id].shape[0]
        if len(self.queues[node]) > 0:
            packet = self.queues[node][0]
            obs = [packet.destination] + [len(self.queues[n]) for n in self.neighbours[node]]
            obs.extend([self.graph[node][n]['delay'] for n in self.neighbours[node]])
            obs.extend([1 if n in packet.visited_nodes else 0 for n in self.neighbours[node]])
            obs.append(len(self.queues[node]))
        else:
            obs = [0] * (1 + len(self.neighbours[node]) * 3 + 1)

        padded_obs = np.zeros(obs_size, dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs

    def perform_action(self, node, action):
        packet_reached_destination = False
        if not self.queues[node]:
            return packet_reached_destination

        num_neighbors = len(self.neighbours[node])
        if num_neighbors == 0:
            return packet_reached_destination

        # 只使用action[0]来选择邻居节点
        primary_neighbor_idx = min(int(action[0] * num_neighbors), num_neighbors - 1)
        primary_neighbor_idx = max(0, primary_neighbor_idx)
        next_node = self.neighbours[node][primary_neighbor_idx]

        # 检查选择的节点是否队列已满
        if next_node in [self.source_node, self.destination_node]:
            initial_remaining_capacity = self.packets - len(self.queues[next_node])
        else:
            initial_remaining_capacity = self.queue_capacity - len(self.queues[next_node])

        # 如果选择的节点队列已满
        if initial_remaining_capacity <= 0:
            # 给予当前节点惩罚
            self.rewards[f"agent_{node}"] += self.rewards_config['queue_full_penalty']

            # 获取所有可用的邻居节点（队列未满的节点）
            available_neighbors = []
            for neighbor in self.neighbours[node]:
                if neighbor in [self.source_node, self.destination_node]:
                    remaining_capacity = self.packets - len(self.queues[neighbor])
                else:
                    remaining_capacity = self.queue_capacity - len(self.queues[neighbor])
                if remaining_capacity > 0:
                    available_neighbors.append(neighbor)

            # 如果有可用的邻居节点，随机选择一个
            if available_neighbors:
                next_node = np.random.choice(available_neighbors)
                if next_node in [self.source_node, self.destination_node]:
                    remaining_capacity = self.packets - len(self.queues[next_node])
                else:
                    remaining_capacity = self.queue_capacity - len(self.queues[next_node])
            else:
                # 如果所有邻居节点都满了，保持在原节点
                return packet_reached_destination

        else:
            remaining_capacity = initial_remaining_capacity

        # 计算可以转发的数据包数量
        available_packets = len(self.queues[node])
        transfer_count = min(available_packets, remaining_capacity)

        # 转发数据包
        for _ in range(transfer_count):
            if not self.queues[node]:
                break
            packet = self.queues[node].popleft()

            # 检查是否重复访问节点，如果是则给予惩罚
            if next_node in packet.visited_nodes:
                self.rewards[f"agent_{node}"] += self.rewards_config['loop_penalty']

            # 检查是否超过最大跳数
            if packet.hops >= packet.max_hops:
                packet.is_dropped = True
                packet.drop_reason = "超过最大跳数限制"
                self.dropped_packets_count += 1  # 增加丢弃计数
                for visited_node in packet.visited_nodes:
                    self.rewards[f"agent_{visited_node}"] += self.rewards_config['max_hops_penalty']
                continue

            # 计算当前节点和下一个节点到目的节点的距离
            current_distance = self.shortest_paths[node]
            next_distance = self.shortest_paths[next_node]

            # 获取边的延迟
            link_delay = self.graph[node][next_node]['delay']
            packet.add_hop(next_node, link_delay)

            # 计算距离相关奖励
            if next_distance < current_distance:
                distance_reward = (current_distance - next_distance) * self.rewards_config['distance_reward_weight']
                self.rewards[f"agent_{node}"] += distance_reward
            else:
                distance_penalty = (next_distance - current_distance) * self.rewards_config['distance_penalty_weight']
                self.rewards[f"agent_{node}"] += distance_penalty

            # 检查是否到达目的地
            if next_node == packet.destination and next_node == self.destination_node:
                packet_reached_destination = True
                self.received_packets_count += 1
                packet.arrival_time = time.time()
                # 对所有参与转发的节点给予完整奖励
                for visited_node in packet.visited_nodes:
                    self.rewards[f"agent_{visited_node}"] += self.rewards_config['packet_arrival_reward']
                continue
            elif next_node != self.destination_node and next_node == packet.destination:
                packet.is_dropped = True
                packet.drop_reason = f"到达错误的目的节点: {next_node}, 应该到达: {self.destination_node}"
                for visited_node in packet.visited_nodes:
                    self.rewards[f"agent_{visited_node}"] += self.rewards_config['wrong_destination_penalty']
                continue

            # 将数据包添加到通道中
            self.channels[node][next_node].append(packet)

        return packet_reached_destination

    def print_episode_summary(self):
        print("\n========== 回合结束，数据包传输统计 ==========")
        print(f"源节点: {self.source_node}, 目的节点: {self.destination_node}")
        print(f"总数据包数: {len(self.current_episode_packets)}, 成功传输数: {self.received_packets_count}")

        # 统计丢包信息
        dropped_packets = [packet_id for packet_id in self.current_episode_packets
                           if self.all_packets[packet_id].is_dropped]
        print(f"丢弃数据包数: {len(dropped_packets)}")

        print("\n数据包ID | 状态 | 转发路径 | 总延迟(ms)/原因")
        print("-" * 70)

        # 修改排序逻辑：按照数字顺序排序
        sorted_packets = sorted(self.current_episode_packets,
                                key=lambda x: int(x.replace('P', '')))

        for packet_id in sorted_packets:
            packet = self.all_packets[packet_id]
            path_str = ' -> '.join(map(str, packet.nodes))
            if packet.is_dropped:
                status = "丢弃"
                info = f"原因: {packet.drop_reason}"
            else:
                if packet.nodes[-1] == self.destination_node:
                    status = "成功"
                    info = f"{packet.total_delay:6.2f}ms"
                else:
                    status = "未完成"
                    info = "未到达目的节点"

            print(f"{packet_id:8} | {status:6} | {path_str:30} | {info}")

        print("=" * 70)

    def check_arrival_milestones(self):
        """检查并奖励到达率里程碑"""
        if self.packets == 0:
            return False
            
        current_arrival_rate = self.received_packets_count / self.packets
        rewards_given = False
        
        for milestone, reward_key in sorted(self.arrival_milestones.items()):
            if milestone not in self.reached_arrival_milestones and current_arrival_rate >= milestone:
                self.reached_arrival_milestones.add(milestone)
                milestone_reward = self.rewards_config[reward_key]
                
                # 给所有智能体发放里程碑奖励
                for agent_id in self.agents:
                    self.rewards[agent_id] += milestone_reward
                
                print(f"\n🎯 到达率达到 {milestone*100}% 里程碑！所有智能体获得奖励：{milestone_reward}")
                rewards_given = True
        
        return rewards_given
