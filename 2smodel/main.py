import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import graph.tools
from twosAGNcode.agcn import Model
from graph.ntu_rgb_d import Graph
import graph.tools
from sklearn.preprocessing import LabelEncoder

# 定义自定义数据集类，用于加载和管理数据和标签
class ActionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        label = self.labels[idx]
        # 如果数据长度小于 300 帧，重复数据直到满 300 帧
        if datum.shape[1] < 300:
            num_repeats = 300 // datum.shape[1]
            remainder = 300 % datum.shape[1]
            repeated_data = np.tile(datum, (num_repeats, 1, 1, 1, 1))
            if remainder > 0:
                partial_data = datum[:remainder]
                repeated_data = np.concatenate((repeated_data, partial_data), axis=0)
            return torch.tensor(repeated_data, dtype=torch.float32), label
        # 如果数据长度大于 300 帧，进行截断
        elif datum.shape[1] > 300:
            return torch.tensor(datum[:300], dtype=torch.float32), label
        else:
            return torch.tensor(datum, dtype=torch.float32), label

# 加载数据和标签
with open(r'E:\guangyichuan\dataset\test\data.pkl', 'rb') as f:
    raw_data = pickle.load(f)
with open(r"E:\guangyichuan\dataset\test\labels.pkl", 'rb') as f:
    raw_labels = pickle.load(f)

# 将字符串标签转换为数字编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(raw_labels)

# 将数据转换为 numpy 数组
data = np.array(raw_data)

new_data_list = []
new_labels_list = []
current_label = None
current_data_chunk = None
current_length = 0
for label, datum in zip(encoded_labels, data):
    if label!= current_label:
        if current_label is not None and current_data_chunk is not None:
            # 将当前数据块转换为五维张量
            data_block_5d = np.expand_dims(current_data_chunk, axis=(0, 4))
            new_data_list.append(data_block_5d)
            new_labels_list.append(current_label)
        current_label = label
        current_data_chunk = [datum]
        current_length = 1
    else:
        current_data_chunk.append(datum)
        current_length += 1

# 处理最后一个数据块
if current_label is not None and current_data_chunk is not None:
    data_block_5d = np.expand_dims(current_data_chunk, axis=(0, 4))
    new_data_list.append(data_block_5d)
    new_labels_list.append(current_label)

# 将数据转换为 PyTorch 张量
new_data_tensor = [torch.tensor(datum, dtype=torch.float32) for datum in new_data_list]
new_labels_tensor = torch.tensor(new_labels_list, dtype=torch.long)

# 创建数据加载器，用于批量加载数据
dataset = ActionDataset(new_data_tensor, new_labels_tensor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 定义模型（使用之前定义的模型）
model = Model(num_class=len(label_encoder.classes_), num_point=11, num_person=1, graph='graph.ntu_rgb_d.Graph', graph_args={})

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将模型移动到 GPU 上（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
for epoch in range(20):
    model.train()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/20, Loss: {loss.item()}')

# 评估模型
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# 推断新数据
def predict(model, data):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())

new_data = np.random.rand(1, 11, 2, 1, 5)  # 模拟的新数据
predictions = predict(model, new_data)
print("Predictions:", predictions)