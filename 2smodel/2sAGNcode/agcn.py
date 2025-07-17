import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# 为了让你的模型能够有效地进行动作识别，需要提供一个经过适当格式化和预处理的数据集。这里是你可以如何准备和使用你的数据集的建议步骤：
# 1. 准备数据集
# 数据格式: 数据集应该是一个五维张量，形状为 (N, C, T, V, M)，分别表示：
# N：批量大小（每次处理的数据样本数量）
# C：通道数（每个关节点的特征数，例如 X, Y, Z 坐标）
# T：时间步长（动作序列中的帧数）
# V：关节点数（每个骨架的关节点数量）
# M：每帧中的人数（通常是 1 或 2）


# 动态导入模块的函数
def import_class(name):
    """
    这个函数用于动态导入指定名称的类。

    参数：
    - name (str): 类的完整路径，例如 'package.module.ClassName'。

    返回：
    - 导入的类对象。
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

# 初始化卷积分支权重的函数
def conv_branch_init(conv, branches):
    """
    初始化卷积层的权重和偏置。

    参数：
    - conv (nn.Conv2d): 要初始化的二维卷积层。
    - branches (int): 分支数量，用于计算权重初始化的值。

    初始化过程：
    - 对于权重，使用正态分布初始化，均值为 0，标准差为 sqrt(2. / (n * k1 * k2 * branches))，其中 n 是权重数量，k1 和 k2 是卷积核的尺寸。
    - 对于偏置，初始化为 0。
    """
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

# 初始化卷积层权重的函数
def conv_init(conv):
    """
    使用 Kaiming 正态分布初始化卷积层的权重。

    参数：
    - conv (nn.Conv2d): 要初始化的二维卷积层。

    Kaiming 初始化通常适用于 ReLU 激活函数的情况，可以帮助模型更快地收敛。
    """
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

# 初始化批量归一化层的函数
def bn_init(bn, scale):
    """
    初始化批量归一化层的权重和偏置。

    参数：
    - bn (nn.BatchNorm2d): 要初始化的二维批量归一化层。
    - scale (float): 权重的初始值。

    初始化过程：
    - 将批量归一化层的权重初始化为 scale。
    - 将偏置初始化为 0。
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

# 定义时序卷积网络单元
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        """
        初始化时序卷积网络单元。

        参数：
        - in_channels (int): 输入通道数。
        - out_channels (int): 输出通道数。
        - kernel_size (int): 卷积核的大小，默认为 9。
        - stride (int): 卷积的步长，默认为 1。

        内部组件：
        - pad: 根据卷积核大小计算的填充值，以保持输出尺寸与输入尺寸相同。
        - conv: 二维卷积层，用于对输入进行时序卷积操作。
        - bn: 批量归一化层，对卷积后的输出进行归一化。
        - relu: ReLU 激活函数。

        初始化过程：
        - 使用 conv_init 函数初始化卷积层的权重和偏置。
        - 使用 bn_init 函数初始化批量归一化层的权重和偏置，初始权重为 1。
        """
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        """
        前向传播过程。

        参数：
        - x (torch.Tensor): 输入张量。

        过程：
        - 先进行卷积操作。
        - 再进行批量归一化。
        - 返回经过处理的张量。
        """
        x = self.bn(self.conv(x))
        return x

# 定义图卷积网络单元
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        """
        初始化图卷积网络单元。

        参数：
        - in_channels (int): 输入通道数。
        - out_channels (int): 输出通道数。
        - A (numpy.ndarray): 邻接矩阵。
        - coff_embedding (int): 嵌入系数，默认为 4。
        - num_subset (int): 子集数量，默认为 3。

        内部组件：
        - inter_channels: 中间通道数，计算为输出通道数除以嵌入系数。
        - PA: 可训练的邻接矩阵参数。
        - A: 不可训练的邻接矩阵，从输入的 numpy 数组转换为 PyTorch 的 Variable。
        - num_subset: 子集数量。
        - conv_a、conv_b、conv_d: 分别是用于不同子集的卷积层列表。
        - down: 如果输入和输出通道数不一致，用于下采样的序列。
        - bn: 批量归一化层。
        - soft: Softmax 激活函数，用于在特定维度上进行归一化。
        - relu: ReLU 激活函数。

        初始化过程：
        - 将输入的邻接矩阵转换为 PyTorch 的 Variable，并设置为不可训练。
        - 创建可训练的邻接矩阵参数，并初始化为一个很小的值。
        - 初始化多个卷积层列表，用于不同的子集。
        - 如果输入和输出通道数不一致，创建下采样层。
        - 初始化批量归一化层、Softmax 激活函数和 ReLU 激活函数。
        - 使用 conv_init 函数初始化所有卷积层的权重和偏置。
        - 使用 bn_init 函数初始化批量归一化层的权重和偏置，初始权重为 1e-6。
        - 使用 conv_branch_init 函数初始化 conv_d 中的卷积层的权重。
        """
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels!= out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        """
        前向传播过程。

        参数：
        - x (torch.Tensor): 输入张量。

        过程：
        - 获取输入张量的尺寸信息。
        - 将邻接矩阵移到与输入张量相同的设备上。
        - 对邻接矩阵进行更新，加上可训练的参数。
        - 对每个子集进行图卷积操作：
            - 首先通过卷积层 conv_a 和 conv_b，然后进行矩阵乘法和 Softmax 归一化，得到新的邻接矩阵 A1。
            - 将输入张量重塑后与新的邻接矩阵进行矩阵乘法，再通过卷积层 conv_d 得到子集的输出 z。
            - 将所有子集的输出累加得到 y。
        - 对 y 进行批量归一化，加上下采样的结果，然后通过 ReLU 激活函数返回。
        """
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

# 定义时序 - 图卷积单元
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        """
        初始化时序 - 图卷积单元。

        参数：
        - in_channels (int): 输入通道数。
        - out_channels (int): 输出通道数。
        - A (numpy.ndarray): 邻接矩阵。
        - stride (int): 时序卷积的步长，默认为 1。
        - residual (bool): 是否使用残差连接，默认为 True。

        内部组件：
        - gcn1: 图卷积网络单元。
        - tcn1: 时序卷积网络单元。
        - relu: ReLU 激活函数。
        - residual: 根据参数决定是否使用残差连接，如果不使用残差连接，将残差连接定义为恒等函数或 0 函数；如果输入和输出通道数相同且步长为 1，则使用恒等函数作为残差连接，否则创建一个时序卷积层作为残差连接。

        初始化过程：
        - 创建图卷积网络单元和时序卷积网络单元。
        - 创建 ReLU 激活函数。
        - 根据参数决定如何设置残差连接。
        """
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        """
        前向传播过程。

        参数：
        - x (torch.Tensor): 输入张量。

        过程：
        - 先通过图卷积网络单元和时序卷积网络单元。
        - 加上残差连接的结果。
        - 通过 ReLU 激活函数返回。
        """
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

# 定义主模型
class Model(nn.Module):
    def __init__(self, num_class=2, num_point=11, num_person=1, graph=None, graph_args=dict(), in_channels=2):
        """
        初始化主模型。

        参数：
        - num_class (int): 分类的类别数量，默认为 60。
        - num_point (int): 关节点数量，默认为 25。
        - num_person (int): 每帧中的人数，默认为 2。
        - graph: 图的类对象或路径。
        - graph_args (dict): 图的参数字典。
        - in_channels (int): 输入通道数，默认为 3。

        内部组件：
        - graph: 根据输入的图类路径动态导入的图类对象。
        - A: 从图对象中获取的邻接矩阵。
        - data_bn: 一维批量归一化层。
        - l1 - l10: 一系列时序 - 图卷积单元。
        - fc: 全连接层。

        初始化过程：
        - 如果输入的图对象为 None，则抛出 ValueError。
        - 否则，根据输入的图类路径动态导入图类对象，并使用图参数创建图对象，获取邻接矩阵 A。
        - 创建一维批量归一化层，用于对输入数据进行归一化。
        - 创建一系列时序 - 图卷积单元，每个单元包含图卷积和时序卷积操作。
        - 创建全连接层，用于最终的分类任务。初始化全连接层的权重和偏置。
        - 使用 bn_init 函数初始化批量归一化层的权重和偏置，初始权重为 1。
        """
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        #定义多个时序-图卷积单元
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        """
        前向传播过程。

        参数：
        - x (torch.Tensor): 输入张量，形状为 (N, C, T, V, M)。

        过程：
        - 获取输入张量的尺寸信息。
        - 调整输入张量的形状，将其重塑为适合后续处理的形式。
        - 通过一维批量归一化层对重塑后的张量进行归一化。
        - 再次调整张量的形状，使其适合通过时序 - 图卷积单元进行处理。
        - 依次通过一系列时序 - 图卷积单元。
        - 获取最后一个时序 - 图卷积单元的输出通道数 c_new。
        """
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

          # 通过多个时序-图卷积单元
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # 平均池化
        c_new = x.size(1)