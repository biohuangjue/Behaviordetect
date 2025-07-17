import numpy as np

def edge2mat(link, num_node):
    """
    将连接列表转换为邻接矩阵的函数。

    参数：
    - link (list): 连接列表，包含节点之间的连接关系。
    - num_node (int): 节点数量。

    返回：
    - A (numpy.ndarray): 邻接矩阵。

    功能：
    - 创建一个全零的方阵，大小为 num_node x num_node。
    - 遍历连接列表，将有连接的节点对应的位置设置为 1。
    """
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    """
    归一化有向图邻接矩阵的函数。

    参数：
    - A (numpy.ndarray): 有向图的邻接矩阵。

    返回：
    - AD (numpy.ndarray): 归一化后的有向图邻接矩阵。

    功能：
    - 计算邻接矩阵 A 的每列的和，存储在 Dl 中。
    - 创建一个与 A 大小相同的全零矩阵 Dn。
    - 对于每列和不为 0 的列，将 Dn 中对应位置的元素设置为该列和的倒数。
    - 计算 A 与 Dn 的矩阵乘积，得到归一化后的有向图邻接矩阵 AD。
    """
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, self_link, inward, outward):
    """
    获取空间图邻接矩阵的函数。

    参数：
    - num_node (int): 节点数量。
    - self_link (list): 自连接列表。
    - inward (list): 向内连接列表。
    - outward (list): 向外连接列表。

    返回：
    - A (numpy.ndarray): 空间图的邻接矩阵，是一个三维数组，包含单位矩阵、归一化后的向内连接矩阵和归一化后的向外连接矩阵。

    功能：
    - 使用 edge2mat 函数创建单位矩阵 I、归一化后的向内连接矩阵 In 和归一化后的向外连接矩阵 Out。
    - 将这三个矩阵堆叠在一起形成一个三维数组作为空间图的邻接矩阵并返回。
    """
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A