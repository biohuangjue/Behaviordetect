import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from agcn import Model
# 定义自定义数据集类，用于加载和管理数据和标签
class ActionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # 存储数据
        self.labels = labels  # 存储标签

    def __len__(self):
        return len(self.data)  # 返回数据集的长度

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # 根据索引返回数据和标签

# 加载数据和标签
with open(r'E:\guangyichuan\dataset\test\data.pkl', 'rb') as f:
    data = pickle.load(f)  # 从data.pkl文件加载数据
with open(r"E:\guangyichuan\dataset\test\labels.pkl", 'rb') as f:
    labels = pickle.load(f)  # 从labels.pkl文件加载标签

# 将字符串标签转换为数字编码
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)  # 将字符串标签转换为数字标签

# 确保数据形状为 (N, C, T, V, M)
# 这里假设你的数据已经是适当的形状，如果不是，需要进行重塑
data = np.array(data)
if data.ndim == 4:
    data = data[:, np.newaxis, :, :, :]  # 添加一个维度 M，使数据形状符合 (N, C, T, V, M)

# 将数据和标签转换为 PyTorch 张量
data_tensor = torch.tensor(data, dtype=torch.float32)  # 转换为浮点数类型的张量
labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)  # 转换为长整型张量

# 创建数据加载器，用于批量加载数据
dataset = ActionDataset(data_tensor, labels_tensor)  # 创建自定义数据集实例
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # 创建数据加载器，batch_size设为8，数据加载时打乱顺序

# 定义模型（使用之前定义的模型）
model = Model(num_class=len(label_encoder.classes_), num_point=11, num_person=1, graph=Graph , graph_args={})

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，学习率设为0.001

# 将模型移动到 GPU 上（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU可用
model.to(device)  # 将模型移动到GPU或CPU

# 训练模型
for epoch in range(20):  # 假设训练 20 轮
    model.train()  # 设置模型为训练模式
    for inputs, labels in dataloader:  # 遍历数据加载器中的数据
        inputs = inputs.to(device)  # 将输入数据移动到GPU或CPU
        labels = labels.to(device)  # 将标签数据移动到GPU或CPU

        optimizer.zero_grad()  # 清除上一步的梯度
        outputs = model(inputs)  # 前向传播，计算模型输出
        loss = criterion(outputs, labels)  # 计算损失值
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

    print(f'Epoch {epoch+1}/20, Loss: {loss.item()}')  # 输出每一轮的损失值

# 评估模型
model.eval()  # 设置模型为评估模式
total_correct = 0
total_samples = 0

with torch.no_grad():  # 禁用梯度计算，提高运行速度，节省内存
    for inputs, labels in dataloader:  # 遍历数据加载器中的数据
        inputs = inputs.to(device)  # 将输入数据移动到GPU或CPU
        labels = labels.to(device)  # 将标签数据移动到GPU或CPU
        outputs = model(inputs)  # 前向传播，计算模型输出
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别（取概率最高的那个）
        total_correct += (predicted == labels).sum().item()  # 累计正确预测的数量
        total_samples += labels.size(0)  # 累计样本总数

accuracy = total_correct / total_samples  # 计算准确率
print(f'Validation Accuracy: {accuracy * 100:.2f}%')  # 输出准确率

# 推断新数据
def predict(model, data):
    model.eval()  # 设置模型为评估模式
    data = torch.tensor(data, dtype=torch.float32).to(device)  # 将数据转换为张量，并移动到GPU或CPU
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(data)  # 前向传播，计算模型输出
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别（取概率最高的那个）
    return label_encoder.inverse_transform(predicted.cpu().numpy())  # 将预测的类别从数字标签转换回原始字符串标签

new_data = np.random.rand(5, 2, 50, 11, 1)  # 模拟的新数据
predictions = predict(model, new_data)  # 使用模型进行推断
print("Predictions:", predictions)  # 输出预测结果
