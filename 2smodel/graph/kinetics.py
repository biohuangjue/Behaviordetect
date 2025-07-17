import numpy as np
import sys
import networkx as nx

# 将父目录添加到模块搜索路径中，以便能从父目录导入模块
sys.path.extend(['../'])

from graph import tools

# 关节点索引常量定义
# 每个索引对应一个身体部位，例如 0 对应 "Nose"
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

# 定义节点数量
num_node = 18

# 自连接列表，每个节点连接到自身
self_link = [(i, i) for i in range(num_node)]

# 向内连接列表，定义了身体部位之间的向内连接关系
inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14)]

# 向外连接列表，是向内连接的反向连接
outward = [(j, i) for (i, j) in inward]

# 邻居连接列表，是向内连接和向外连接的总和
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        """
        Graph 类的构造函数。

        参数：
        - labeling_mode (str): 标记模式，默认为 'spatial'。

        功能：
        - 根据传入的标记模式获取邻接矩阵 A，并存储图的相关属性，如节点数量、自连接、向内连接、向外连接和邻居连接。
        """
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        """
        获取邻接矩阵的方法。

        参数：
        - labeling_mode (str): 标记模式，如果为 None，则返回当前的邻接矩阵 A。

        返回：
        - 根据标记模式构建的邻接矩阵。

        功能：
        - 如果标记模式为 None，直接返回当前的邻接矩阵 A。
        - 如果标记模式为 'spatial'，调用 tools.get_spatial_graph 函数构建空间图的邻接矩阵并返回。
        - 如果标记模式不是 'spatial'，抛出 ValueError。
        """
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    """
    主程序入口，用于测试 Graph 类。
    创建一个 Graph 对象，并打印其邻接矩阵。
    """
    A = Graph('spatial').get_adjacency_matrix()
    print('')