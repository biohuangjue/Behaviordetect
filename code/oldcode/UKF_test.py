import cv2
import sleap
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor  # 用于预测缺失的点
import random
import pickle
import math
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense ,Dropout # 确保导入 Dense 层


# 初始化 RandomForest 分类器
rf = RandomForestClassifier()  # 您可以根据需要设置参数
data_current=[]
class_data_0 = []  # 记录标签为 0 的数据
class_data_1 = []  # 记录标签为 1 的数据
label_state = 0  # 用于记录当前输入数据的标签

# 存储数据集与标签
data = []
data=np.array(data)
labels = []

# 加载 sleap 模型
predictor = sleap.load_model(["F:/guangyichuan/models/240727_145201.centroid.n=38.zip", "F:/guangyichuan/models/240727_150142.centered_instance.n=38.zip"], batch_size=16)  # 采用的 sleap 模型

x1 = None  # 定义全局变量
point_data = None  # 用于存储点的数据


def dst(a, b, c, d):  # 计算距离、速度
    v = ((a - c) ** 2 + (b - d) ** 2) ** (1 / 2)
    return v

class UnscentedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, alpha=0.1, beta=2):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.alpha = alpha
        self.beta = beta

        # 计算西格玛点的参数
        self.n = len(initial_state)
        self.kappa = 0
        self.lambda_ = self.n + self.kappa

        # 计算权重
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - math.pow(self.alpha, 2) + self.beta)
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.n + self.lambda_))

    def f(self,state, control_input):
        # 这里简单假设一个非线性运动模型
        x = state[0] + control_input[0] * np.cos(state[1])
        y = state[1] + control_input[0] * np.sin(state[1])
        return np.array([x, y])

    def h(self,state):
        # 假设测量模型为直接返回状态
        return state

    def generate_sigma_points(self):
        # 计算平方根矩阵
        root_cov = np.linalg.cholesky((self.n + self.lambda_) * self.covariance)
        # 生成西格玛点
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state
        for i in range(self.n):
            sigma_points[i + 1] = self.state + root_cov[:, i]
            sigma_points[self.n + i + 1] = self.state - root_cov[:, i]
        return sigma_points

    def predict(self, control_input):
        # 生成西格玛点
        sigma_points = self.generate_sigma_points()
        # 传播西格玛点
        propagated_sigma_points = np.array([self.f(point, control_input) for point in sigma_points])
        # 计算预测状态
        predicted_state = np.dot(self.Wm, propagated_sigma_points)
        # 计算预测协方差
        predicted_covariance = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = propagated_sigma_points[i] - predicted_state
            predicted_covariance += self.Wc[i] * np.outer(diff, diff)
        predicted_covariance += self.process_noise_cov
        self.state = predicted_state
        self.covariance = predicted_covariance

    def update(self, measurement):
        # 生成预测的西格玛点
        predicted_sigma_points = self.generate_sigma_points()
        # 计算预测的测量值
        predicted_measurements = np.array([self.h(point) for point in predicted_sigma_points])
        # 计算测量的均值
        predicted_measurement_mean = np.dot(self.Wm, predicted_measurements)
        # 计算创新协方差
        innovation_covariance = np.zeros((len(measurement), len(measurement)))
        for i in range(2 * self.n + 1):
            diff = predicted_measurements[i] - predicted_measurement_mean
            innovation_covariance += self.Wc[i] * np.outer(diff, diff)
        innovation_covariance += self.measurement_noise_cov
        # 计算交叉协方差
        cross_covariance = np.zeros((self.n, len(measurement)))
        for i in range(2 * self.n + 1):
            diff_state = predicted_sigma_points[i] - self.state
            diff_measurement = predicted_measurements[i] - predicted_measurement_mean
            cross_covariance += self.Wc[i] * np.outer(diff_state, diff_measurement)
        # 计算卡尔曼增益
        kalman_gain = np.dot(cross_covariance, np.linalg.inv(innovation_covariance))
        # 更新状态估计
        innovation = measurement - predicted_measurement_mean
        self.state = self.state + np.dot(kalman_gain, innovation)
        # 更新协方差估计
        self.covariance = self.covariance - np.dot(np.dot(kalman_gain, innovation_covariance), kalman_gain.T)

# 定义包含优化后的 RNN 模型的类
class RNNModel:
    def __init__(self):
        self.model = Sequential([
            SimpleRNN(units=128, activation='relu', input_shape=(11, 2)),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, data, labels, epochs=10):
        self.model.fit(data, labels, epochs=epochs)

    def predict(self, new_input_data):
        return self.model.predict(new_input_data)


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


def draw_point(v, frame,a,b,c):
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
ukf = [UnscentedKalmanFilter(np.array([0, 0]), np.eye(2), np.array([[0.1, 0], [0, 0.1]]), np.array([[0.01, 0], [0, 0.01]])) for _ in range(11)]

def analyze_frames():
    global frame_count, label_state ,data ,labels
    width = 1280
    height = 1024
    cap = cv2.VideoCapture("F:/guangyichuan/videos/0702Sweet 5+5+5-1.mp4")  # 视频读取路径
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置比例
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    vll = None
    frame_count = 0  # 已记录帧数
    output_path = "F:/new/new_test_videos/new_1.mp4"  # 输出路径
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出格式
    out = cv2.VideoWriter(output_path, fourcc, 90, (width, height))
    a, b, c = 1, 5, 6  # 胸，左前腿，右前腿
    vll_past = None  # 初始化前一帧
    folder_name = input("请输入文件夹名称: ")  # 让用户输入文件夹名称
    load_data(folder_name)
    
    def on_button_click(event, x, y, flags, param):
        global label_state
        if event == cv2.EVENT_LBUTTONDOWN:
            if label_state == 0:
                label_state = 1
            else:
                label_state = 0

    # 创建优化后的 RNN 模型对象并训练
    data=np.array(data)
    data = np.reshape(data, (data.shape[0], 11, 2))
    labels=np.array(labels)

    print(labels.shape)
    print(data.shape)
    rnn_model= RNNModel()
    rnn_model.fit(data, labels,epochs=10)

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
            vll=np.nan_to_num(vll, nan=0)
            print(vll)
            if vll.shape[1] == 1 and vll_past is None:  # 第一次有数据时初始化 vll_past
                vll_past = vll.copy()
            if vll.shape[1] == 1:  # 有数据

                
                #x1, y1 = vll[0][0][a][0], vll[0][0][a][1]
                #x2, y2 = vll[0][0][b][0], vll[0][0][b][1]
                #x3, y3 = vll[0][0][c][0], vll[0][0][c][1]
                if vll_past.shape[1] == 1:
                    
                    control_input = np.array([0.1, 0])  # 假设的控制输入
                    for i in range(11):
                        ukf[i].predict(control_input)
                        if vll[0][0][i][0]!= 0 and vll[0][0][i][1]!= 0:  # 检查点是否不为 0
                            ukf[i].update(np.array([vll[0][0][i][0], vll[0][0][i][1]]))  # 进行 update 操作
                            
                        if vll[0][0][i][0] == 0:                          
                            vll[0][0][i][0], vll[0][0][i][1] = ukf[i].state[0], ukf[i].state[1]  
                data_current.append(vll[0][0])

                    #x1_past, y1_past = vll_past[0][0][a][0], vll_past[0][0][a][1]
                    #d1, v1 = dst(x2, y2, x3, y3), dst(x1, y1, x1_past, y1_past)
                draw_point(vll, frame,0,255,0)

                if frame_count>50:
                    vll_reshaped = np.reshape(vll[0][0], (1, 11, 2))
                    prediction = rnn_model.predict(vll_reshaped)  
                    print(prediction)
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
    

analyze_frames()