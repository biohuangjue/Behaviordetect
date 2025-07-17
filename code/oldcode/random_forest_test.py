import cv2
import sleap
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor#用于预测缺失的点
import random
import pickle

# 初始化 RandomForest 分类器
rf = RandomForestClassifier()  # 您可以根据需要设置参数

class_data_0 = []#记录标签为0的数据
class_data_1 = []#记录标签为1的数据
label_state = 0#用于记录当前输入数据的标签

# 存储数据集与标签
data = []
labels = []

# 加载 sleap 模型
predictor = sleap.load_model(["F:/new/models/240703_172214.centroid.n=86.zip", "F:/new/models/240703_174034.centered_instance.n=86.zip"], batch_size=16)#采用的sleap模型

x1 = None  # 定义全局变量
point_data = None  # 用于存储点的数据

def update_data(a):  # 更新训练集，100个数据点之后随机更换
    global data, labels, label_state
    new_data = a
    data.append(new_data)
    if label_state == 0:
        if len(class_data_0) < 100:
            class_data_0.append(new_data)
            labels.append(label_state)
            rf.fit(class_data_0 + class_data_1, labels)
        else:
            # 随机替换一个已有数据
            index = random.randint(0, 99)
            class_data_0[index] = new_data
            rf.fit(class_data_0 + class_data_1, labels)
    elif label_state == 1:
        if len(class_data_1) < 100:
            class_data_1.append(new_data)
            labels.append(label_state)
            rf.fit(class_data_0 + class_data_1, labels)
        else:
            # 随机替换一个已有数据
            index = random.randint(0, 99)
            class_data_1[index] = new_data
            rf.fit(class_data_0 + class_data_1, labels)


def dst(a, b, c, d):  # 计算距离、速度
    v = ((a - c) ** 2 + (b - d) ** 2) ** (1 / 2)
    return v


def save_data():
    """保存数据集和标签"""
    with open("C:/Users/A/Desktop/guangyichuan/dataset/cuoshou/0.pkl", 'wb') as f0:
        pickle.dump(class_data_0, f0)
    with open("C:/Users/A/Desktop/guangyichuan/dataset/cuoshou/1.pkl", 'wb') as f1:
        pickle.dump(class_data_1, f1)
    with open("C:/Users/A/Desktop/guangyichuan/dataset/cuoshou/labels.pkl", 'wb') as f2:
        pickle.dump(labels, f2)

def load_data():
    """加载数据集和标签"""
    global class_data_0, class_data_1, labels
    try:
        with open("C:/Users/A/Desktop/guangyichuan/dataset/cuoshou/0.pkl", 'rb') as f0:
            class_data_0 = pickle.load(f0)
    except FileNotFoundError:
        class_data_0 = []
    try:
        with open("C:/Users/A/Desktop/guangyichuan/dataset/cuoshou/1.pkl", 'rb') as f1:
            class_data_1 = pickle.load(f1)
    except FileNotFoundError:
        class_data_1 = []
    try:
        with open("C:/Users/A/Desktop/guangyichuan/dataset/cuoshou/labels.pkl", 'rb') as f2:
            labels = pickle.load(f2)
    except FileNotFoundError:
        labels = []


def draw_point(v, frame):
    """在给定的帧上绘制点"""
    global point_data
    if v.shape[1] == 1:  # 有新的点数据
        point_data = v[0][0]  # 更新点数据
    if point_data is not None:
        for point in point_data:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # 在图像上绘制点



def analyze_frames():
    global frame_count, label_state
    width = 1280
    height = 1024
    cap = cv2.VideoCapture("F:/guangyichuan/videos/gr66c 3.7.1-1.mp4")#视频读取路径
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置比例
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    vll=None
    frame_count = 0  # 已记录帧数
    output_path = "F:/new/new_test_videos/new_1.mp4"  # 输出路径
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出格式
    out = cv2.VideoWriter(output_path, fourcc, 90, (width, height))
    a, b, c = 1, 5, 6  # 胸，左前腿，右前腿
    vll_past = None  # 初始化前一帧
    
    load_data()
    def on_button_click(event, x, y, flags, param):
        global label_state
        if event == cv2.EVENT_LBUTTONDOWN:
            if label_state == 0:
                label_state = 1
            else:
                label_state = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

        out.write(frame)
        frame_count += 1

        if frame_count % 1 == 0:  # 每 50 帧绘制点
            frame_predictions = predictor.inference_model.predict_on_batch(np.expand_dims(frame, axis=0))
            vl = frame_predictions['instance_peaks']
            vll = vl.numpy()
            vll = np.nan_to_num(vll, nan=0)
            vll = vll.astype(np.float64)
            if vll.shape[1] == 1 and vll_past is None:  # 第一次有数据时初始化 vll_past
                vll_past = vll.copy()
            if vll.shape[1] == 1:  # 有数据
                print(vll)
                x1, y1 = vll[0][0][a][0], vll[0][0][a][1]
                x2, y2 = vll[0][0][b][0], vll[0][0][b][1]
                x3, y3 = vll[0][0][c][0], vll[0][0][c][1]
                if vll_past.shape[1] == 1:
                    mask = (vll == 0)  # 找出 vll 中的 0 位置
                    
                    vll[mask] = vll_past[mask]  # 用 vll_past 中对应位置的值填充 vll 中的 NaN
                    x1_past, y1_past = vll_past[0][0][a][0], vll_past[0][0][a][1]
                    d1, v1 = dst(x2, y2, x3, y3), dst(x1, y1, x1_past, y1_past)
                    print(vll_past)
                    
                    
                    if frame_count > 2000:  # 开始 predict 的时间
                        print(rf.predict([[d1, v1]]))
                    update_data([d1, v1])
                    print(label_state)
        draw_point(vll, frame)  # 在 drawn_frame 上绘制点

        cv2.imshow('Camera Feed', frame)  # 始终显示绘制了点的帧

        # 注册鼠标点击事件回调函数
        cv2.setMouseCallback('Camera Feed', on_button_click)
        key = cv2.waitKey(1)
        if key == 113:  # 按 q 可以终止程序
            break

        if frame_count >= 50 and vll_past is not None:  # 如果有 50 帧前的数据且不为空
            # 在此处进行当前 vll 和 50 帧前的 vll_past 的比较或计算
            vll_past = vll.copy()  # 存储当前 vll 作为 50 帧前的数据

    cap.release()
    save_data()

analyze_frames()