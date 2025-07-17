import sleap
import numpy as np
import cv2

import random
import pickle
import math

import time
import ukf_module  # 导入刚才创建的模块
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class_data_0 = []  # 记录标签为 0 的数据
class_data_1 = []  # 记录标签为 1 的数据
label_state = 0  # 用于记录当前输入数据的标签

# 存储数据集与标签
data = []
labels = []

# 加载 sleap 模型
predictor = sleap.load_model(["E:/guangyichuan/models/240727_145201.centroid.n=38.zip", "E:/guangyichuan/models/240727_150142.centered_instance.n=38.zip"], batch_size=10)  # 采用的 sleap 模型

def update_data(a):  # 更新训练集，100 个数据点之后随机更换
    global data, labels, label_state
    if label_state==1:
        class_data_1.append(a)
    if label_state==0:
        class_data_0.append(a)
    data.append(a)
    labels.append(label_state)

def save_data(folder_name):
    """保存数据集和标签"""
    with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/0.pkl", 'wb') as f0:
        pickle.dump(class_data_0, f0)
    with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/1.pkl", 'wb') as f1:
        pickle.dump(class_data_1, f1)
    with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/labels.pkl", 'wb') as f2:
        pickle.dump(labels, f2)
    with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/data.pkl", 'wb') as f3:
        pickle.dump(data, f3)


def load_data(folder_name):
    """加载数据集和标签"""
    global class_data_0, class_data_1, labels, data
    try:
        with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/0.pkl", 'rb') as f0:
            class_data_0 = pickle.load(f0)
    except FileNotFoundError:
        class_data_0 = []
    try:
        with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/1.pkl", 'rb') as f1:
            class_data_1 = pickle.load(f1)
    except FileNotFoundError:
        class_data_1 = []
    try:
        with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/labels.pkl", 'rb') as f2:
            labels = pickle.load(f2)
    except FileNotFoundError:
        labels = []
    try:
        with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/data.pkl", 'rb') as f3:
            data = pickle.load(f3)
    except FileNotFoundError:
        data = []
        print("filenotfound")


def draw_point(v, frame, a, b, c):
    """在给定的帧上绘制点"""
    global point_data
    if v.shape[1] == 1:  # 有新的点数据
        point_data = v[0][0]  # 更新点数据
    if point_data is not None:
        for point in point_data:
            x, y = point
            x = min(max(x, 0), frame.shape[1] - 1)  # 限制 x 坐标在合理范围内
            y = min(max(y, 0), frame.shape[0] - 1)  # 限制 y 坐标在合理范围内
            cv2.circle(frame, (int(x), int(y)), 5, (a, b, c), -1)  # 在图像上绘制点


# 初始化 UKF
ukf = [ukf_module.UnscentedKalmanFilter(np.array([0, 0]), np.eye(2), np.array([[0.1, 0], [0, 0.1]]), np.array([[0.01, 0], [0, 0.01]])) for _ in range(11)]


def analyze_frames():
    global frame_count, label_state, data, labels
    width = 1280
    height = 1024
    start_frame = int(input("请输入开始的帧："))
    cap = cv2.VideoCapture(r"E:\guangyichuan\videos\dataset_onedrosophila\gr5a2.7.3-1.mp4")  # 视频读取路径
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置比例
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 新增：设置倍速，例如 2 倍速
    playback_speed = 5  # 您可以根据需要修改此值
    cap.set(cv2.CAP_PROP_FPS, 90)

    vll = None
    frame_count = 0  # 已记录帧数
    folder_name = input("请输入文件夹名称: ")  # 让用户输入文件夹名称

    load_data(folder_name)  # 传入用户输入的文件夹名称

    def on_button_click(event, x, y, flags, param):
        global label_state
        if event == cv2.EVENT_LBUTTONDOWN and x < 100 and y < 100:  # 检查是否点击了左上角 100x100 的区域
            if label_state == 0:
                label_state = 1
            else:
                label_state = 0

    prev_time = time.time()  # 记录上一帧的时间

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 计算并显示帧率
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"Label State: {label_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 显示 label_state
        cv2.putText(frame, f"Data Length: {len(data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 显示 data 长度
        cv2.putText(frame, f"Class Data 0 Length: {len(class_data_0)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 显示 class_data_0 长度
        cv2.putText(frame, f"Class Data 1 Length: {len(class_data_1)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 显示 class_data_1 长度
        # 绘制大按钮
        cv2.rectangle(frame, (0, 0), (100, 100), (0, 0, 255), 3)  # 绘制一个 100x100 的红色矩形按钮
        if frame_count % 1 == 0:  # 每 50 帧绘制点
            frame_predictions = predictor.inference_model.predict_on_batch(np.expand_dims(frame, axis=0))
            vl = frame_predictions['instance_peaks']
            vll = vl.numpy()
            vll = np.nan_to_num(vll, nan=0)
            vll_past = None

            if vll.shape[1] == 1 and vll_past is None:  # 第一次有数据时初始化 vll_past
                vll_past = vll.copy()
            if vll.shape[1] == 1:  # 有数据
                if frame_count > 500:
                    update_data(vll[0][0])
                if vll_past.shape[1] == 1:
                    control_input = np.array([0.1, 0])  # 假设的控制输入
                    for i in range(11):
                        ukf[i].predict(control_input)
                        if vll[0][0][i][0]!= 0 and vll[0][0][i][1]!= 0:  # 检查点是否不为 0
                            ukf[i].update(np.array([vll[0][0][i][0], vll[0][0][i][1]]))  # 进行 update 操作

                        if vll[0][0][i][0] == 0:
                            vll[0][0][i][0], vll[0][0][i][1] = ukf[i].state[0], ukf[i].state[1]
                draw_point(vll, frame, 0, 255, 0)
        cv2.imshow('Camera Feed', frame)  # 始终显示绘制了点的帧
        cv2.setMouseCallback('Camera Feed', on_button_click)
        key = cv2.waitKey(1)
        if key == 113:  # 按 q 可以终止程序
            break

        if frame_count >= 50 and vll_past is not None:  # 如果有 50 帧前的数据且不为空
            # 在此处进行
            vll_past = vll.copy()  # 存储当前 vll 作为 50 帧前的数据
    cap.release()
    save_data(folder_name)  # 传入用户输入的文件夹名称

analyze_frames()