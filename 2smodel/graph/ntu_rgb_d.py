import sys

# 将父目录添加到模块搜索路径中，以便能从父目录导入模块
sys.path.extend(['../'])
from graph import tools# type: ignore

# 定义节点数量
num_node = 11

# 自连接列表，每个节点连接到自身
self_link = [(i, i) for i in range(num_node)]

# 原始向内连接索引列表
inward_ori_index = [(2,6),(3,7),(4,8),(4,9),(5,10),(5,11),(4,5),(1,2),(1,3),(4,1)]#大致试了一下果蝇

# 向内连接列表，将原始索引转换为从 0 开始的索引
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]

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
    主程序入口，用于测试 Graph 类并可视化邻接矩阵。

    功能：
    - 导入 matplotlib.pyplot 和 os 模块。
    - 创建一个 Graph 对象并获取其邻接矩阵。
    - 对于每个邻接矩阵，使用 matplotlib 进行可视化显示。
    - 最后打印邻接矩阵。
    """
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)