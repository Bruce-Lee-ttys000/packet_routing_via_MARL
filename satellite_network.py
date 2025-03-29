import networkx as nx
import math


def calculate_distance(height, angle):
    """计算两个卫星之间的距离
    
    Args:
        height (float): 卫星高度（公里）
        angle (float): 两个卫星之间的角度（弧度）
        
    Returns:
        float: 两个卫星之间的距离（公里）
    """
    earth_radius = 6371  # 地球半径（公里）
    return math.sqrt(
        (earth_radius + height) ** 2 + (earth_radius + height) ** 2 - 2 * (
                earth_radius + height) ** 2 * math.cos(angle))


def find_farthest_nodes(graph):
    """找到图中最远的两个节点
    
    Args:
        graph (networkx.Graph): 网络图
        
    Returns:
        tuple: (源节点, 目标节点)
    """
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    max_hops = 0
    src_node = None
    dst_node = None

    for src in lengths:
        for dst in lengths[src]:
            if lengths[src][dst] > max_hops:
                max_hops = lengths[src][dst]
                src_node = src
                dst_node = dst

    return src_node, dst_node


def create_satellite_network(config):
    """创建田字型卫星网络拓扑

    Args:
        config: 配置对象，包含以下属性：
            - satellite_height: 卫星高度（公里）
            - grid_size: 田字型网格大小，默认为3

    Returns:
        tuple: (networkx.Graph, source_node, destination_node)
    """
    # 获取网格大小，默认为3x3
    grid_size = getattr(config, 'grid_size', 5)
    total_satellites = grid_size * grid_size
    G = nx.Graph()

    # 添加卫星节点
    for k in range(total_satellites):
        G.add_node(k, height=config.satellite_height)

    # 创建田字型拓扑
    # 横向连接
    for i in range(grid_size):
        for j in range(grid_size - 1):
            node1 = i * grid_size + j
            node2 = i * grid_size + j + 1
            # 计算距离和延迟
            angle = math.pi / (4 * grid_size)  # 假设的角度
            distance = calculate_distance(config.satellite_height, angle)
            delay = (distance / 299792.458) * 1000  # 光速转换为毫秒
            G.add_edge(node1, node2, delay=delay)

    # 纵向连接
    for i in range(grid_size - 1):
        for j in range(grid_size):
            node1 = i * grid_size + j
            node2 = (i + 1) * grid_size + j
            # 计算距离和延迟
            angle = math.pi / (4 * grid_size)  # 假设的角度
            distance = calculate_distance(config.satellite_height, angle)
            delay = (distance / 299792.458) * 1000  # 光速转换为毫秒
            G.add_edge(node1, node2, delay=delay)

    # 固定源节点为左上角(0)，目的节点为右下角(grid_size*grid_size-1)
    source_node = 0
    destination_node = total_satellites - 1

    # 打印网络拓扑信息
    print(f"\n创建了田字型卫星网络拓扑:")
    print(f"网格大小: {grid_size}x{grid_size}, 总卫星数: {total_satellites}")
    print(f"源节点: {source_node} (左上角), 目的节点: {destination_node} (右下角)")

    return G, source_node, destination_node
