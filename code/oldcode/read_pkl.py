import pickle

# 加载pkl文件中的列表
with open(r"E:\guangyichuan\videos\dataset_onedrosophila\data\0702Sweet5+5+5-1.mp4_data.pkl", 'rb') as f:
    data_list = pickle.load(f)

# 打印列表的前十个元素
print("前十个元素：")
for i in range(min(10, len(data_list))):
    print(data_list[i])

# 输出列表的形状
print("\n列表的形状：", np.array(data_list).shape)
