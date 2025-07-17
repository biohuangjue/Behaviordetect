import cv2
import sleap
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier

# 初始化 KNN 分类器
datalen = 1
knn = KNeighborsClassifier(n_neighbors=20)
datalen -= 1
label_state = 0
# 存储数据集
data = []
labels = []
# 加载 sleap 模型
predictor = sleap.load_model(["F:/new/models/240703_172214.centroid.n=86.zip", "F:/new/models/240703_174034.centered_instance.n=86.zip"], batch_size=16)

x1 = None  # 定义全局变量

def update_data(a):  # 更新训练集
    global data, labels, label_state, datalen
    new_data = a
    data.append(new_data)
    labels.append(label_state)
    knn.fit(data, labels)
    datalen += 1

def dst(a, b, c, d):  # 计算距离、速度
    v = ((a - c) ** 2 + (b - d) ** 2) ** (1 / 2)
    return v

def analyze_frames():
    global frame_count, label_state
    width = 1280
    height = 1024
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置比例
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame_count = 0  # 已记录帧数
    output_path = "F:/new/new_test_videos/new_1.mp4"  # 输出路径
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出格式
    out = cv2.VideoWriter(output_path, fourcc, 90, (width, height))  
    a, b, c = 1, 5, 6  # 胸，左前腿，右前腿
    vll_past = None  # 初始化前一帧 

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
        cv2.imshow('Camera Feed', frame)

        # 注册鼠标点击事件回调函数
        cv2.setMouseCallback('Camera Feed', on_button_click)
        key=cv2.waitKey(1)
        if frame_count % 50 == 0:  # 每 50 帧取 1 帧
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
                    print(x1)
                    print(d1, v1)
                    if frame_count>2000: #开始predict的时间
                        print(knn.predict([[d1, v1]]))
                    update_data([d1, v1])
                    print(label_state)
                    
            if frame_count >= 50 and vll_past is not None:  # 如果有 50 帧前的数据且不为空
                # 在此处进行当前 vll 和 50 帧前的 vll_past 的比较或计算
                vll_past = vll.copy()  # 存储当前 vll 作为 50 帧前的数据

    cap.release()

analyze_frames()