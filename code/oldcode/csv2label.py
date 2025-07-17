import pandas as pd
import pickle

# 读取CSV文件
df = pd.read_csv(r"E:\guangyichuan\videos\dataset_onedrosophila\try5.csv")

# 获取最大的 Image index 以确定事件列表的长度
max_index = df['Image index'].max()

# 初始化事件列表，长度为最大 Image index 加 1，默认值为 'normal'
event_list = ['normal'] * (max_index + 1)

# 初始化事件字典
events = {}

# 遍历CSV文件，解析事件的开始和结束
for i, row in df.iterrows():
    behavior = row['Behavior']
    behavior_type = row['Behavior type']
    image_index = row['Image index']

    if behavior_type == 'START':
        events[behavior] = image_index
    elif behavior_type == 'STOP':
        start_index = events.pop(behavior, None)
        if start_index is not None:
            for j in range(start_index, image_index + 1):
                event_list[j] = behavior

# 生成降采样后的列表，只保留能整除5的元素
downsampled_event_list = [event_list[i] for i in range(len(event_list)) if i % 5 == 0]

# 保存降采样后的事件列表为pickle文件
with open(r"E:\guangyichuan\videos\dataset_onedrosophila\downsampled_event_list.pkl", 'wb') as f:
    pickle.dump(downsampled_event_list, f)

print("降采样后的事件列表已保存为 downsampled_event_list.pkl")
