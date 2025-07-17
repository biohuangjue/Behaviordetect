import cv2
import sleap
import pywt
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor#用于预测缺失的点
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random
import pickle
import matplotlib.pyplot as plt
# 加载 sleap 模型
predictor = sleap.load_model(["F:/new/models/240703_172214.centroid.n=86.zip", "F:/new/models/240703_174034.centered_instance.n=86.zip"], batch_size=16)#采用的sleap模型
bodypart_selected = [1, 5, 6]  # 胸，左前腿，右前腿
data_window = np.empty((0,6))
bodypart_index = []
tot_features = np.empty((0,6))
sampling_frequency = 90
step = 10
window_size = 60

def extract_wavelet_features_with_window(sig, sampling_frequency, step,wavelet='cmor1.5-2.0', 
                                         maxfreq=90):
    global window_size
    f_range = np.arange(0.1, maxfreq, maxfreq / 100)  # Frequency range
    wav = pywt.ContinuousWavelet(wavelet)
    fc = wav.center_frequency
    widths = (sampling_frequency * fc) / f_range

    features = []
    for i in range(0, len(sig) - window_size + 1, step):
        window_sig = sig[i:i + window_size]
        coef, freqs = pywt.cwt(window_sig, widths, wavelet, 1 / sampling_frequency)
        power = np.log2(abs(coef))
        reduced_power = np.mean(power, axis=0)
        features.append(reduced_power)

    # 填充以匹配原始信号长度
    features = np.array(features)
    print(features.shape)
    print(features)
    print(sig.shape)
    return features

def process_body_part(part, aligned_df, sampling_frequency):
    part_x = aligned_df[:,part]
    part_y = aligned_df[:,part+1]

    # 提取X坐标的小波特征
    
    part_x_features = extract_wavelet_features_with_window(part_x,  sampling_frequency,step)
    
    # 提取Y坐标的小波特征
    part_y_features = extract_wavelet_features_with_window(part_y, sampling_frequency,step )

    return part_x_features, part_y_features

def update_data(a):  # 更新小波特征数据
    global data_window, window_size
    new_data = a
    data_window = np.append(data_window,new_data,axis=0)
    print('updated data',data_window)

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
    global frame_count, tot_features, sampling_frequency, step, bodypart_selected,data_window 
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
    
    vll_past = None  # 初始化前一帧
    prev_time = time.time()


    while True:
        ret, frame = cap.read()
        if not ret:
            break


        out.write(frame)
        frame_count += 1
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if frame_count % 1 == 0:  # 每 20 帧绘制点
            frame_predictions = predictor.inference_model.predict_on_batch(np.expand_dims(frame, axis=0))
            vl = frame_predictions['instance_peaks']
            vll = vl.numpy()
            vll = np.nan_to_num(vll, nan=0)
            vll = vll.astype(np.float64)
            if vll.shape[1] == 1 and vll_past is None:  # 第一次有数据时初始化 vll_past
                vll_past = vll.copy()
            if vll.shape[1] == 1:  # 有数据
                if vll_past.shape[1] == 1:
                    mask = (vll == 0)  # 找出 vll 中的 0 位置
                    vll[mask] = vll_past[mask]  # 用 vll_past 中对应位置的值填充 vll 中的 NaN
                    print(vll_past)
                
                print(vll)
                xy_coord = np.empty((0,))
                xy_features_parts = np.empty((0,))
                for j in range(len(bodypart_selected)):
                    x1, y1 = vll[0][0][bodypart_selected[j]][0], vll[0][0][bodypart_selected[j]][1]
                    
                    xy_coord=np.append(xy_coord,[x1,y1])
                    
                xy_coord = xy_coord.reshape(1,2*len(bodypart_selected))
                update_data(xy_coord)
            draw_point(vll, frame)  # 在 drawn_frame 上绘制点
            cv2.imshow('Camera Feed', frame)  # 始终显示绘制了点的帧
            key = cv2.waitKey(1)
            if key == 113:  # 按 q 可以终止程序
                break
            if frame_count >= 50 and vll_past is not None:  # 如果有 50 帧前的数据且不为空
            # 在此处进行当前 vll 和 50 帧前的 vll_past 的比较或计算
                vll_past = vll.copy()  # 存储当前 vll 作为 50 帧前的数据
    cap.release()
    arrays = []
    for j in range(len(bodypart_selected)):
        mono_x_feature,mono_y_feature = process_body_part(j,data_window,sampling_frequency)
        xy_features_onepart = np.concatenate((mono_x_feature,mono_y_feature),axis=1)
        arrays.append(xy_features_onepart)
    tot_features = np.concatenate(arrays,axis=1)
    print(tot_features.shape)
                    
                    
                    
            

        


        



    
    
    #save_data()
def cluster_plot(data, nc): #data: columns as features, rows as samples
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)
    pca = PCA(n_components=nc)
    pca_result = pca.fit_transform(scaled_features)
    # 尝试不同的K值
    K_values = range(1, 11)
    wcss = []
    d_wcss = []
    d2_wcss = []
    for k in K_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)  # 存储每个K值的WCSS
        if(k>1):
            d_wcss.append(wcss[k-1]-wcss[k-2])
            if(k>2):
                d2_wcss.append(d_wcss[k-2]-d_wcss[k-3])
    
    optimal_k = d2_wcss.index(max(d2_wcss)) +2
    
    # 绘制肘部图
    plt.figure(figsize=(8, 6))
    plt.plot(K_values, wcss, '-o')
    plt.xlabel('Number of clusters, K')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method For Optimal K')
    plt.show()
    print(optimal_k)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    # 拟合模型
    kmeans.fit(scaled_features)

    # 预测每个点的簇标签
    predicted_labels = kmeans.predict(scaled_features)

    # 查看簇中心
    cluster_centers = kmeans.cluster_centers_


    # 绘制降维后的3D点图
    fig = plt.figure(figsize=(8, 6))
    #ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(predicted_labels)
    for label in unique_labels:
        # 根据聚类标签筛选数据点
        cluster_mask = (predicted_labels == label)
        plt.scatter(pca_result[cluster_mask, 0], pca_result[cluster_mask, 1], label=f'Cluster {label}')
    plt.show()
    return predicted_labels
def save_data(folder_name, data_name):
    """保存数据集"""
    with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/{data_name}.pkl", 'wb') as f3:
        pickle.dump(tot_features, f3)


def load_data(folder_name, data_name):
    """加载数据集和标签"""
    global tot_features
    try:
        with open(f"C:/Users/A/Desktop/guangyichuan/dataset/{folder_name}/{data_name}.pkl", 'rb') as f3:
            tot_features = pickle.load(f3)
    except FileNotFoundError:
        data = []
        print("filenotfound")
def label_video(predicted_labels):
    global frame_count, tot_features, sampling_frequency, step, bodypart_selected,data_window 
    width = 1280
    height = 1024
    cap = cv2.VideoCapture("F:/guangyichuan/videos/gr66c 3.7.1-1.mp4")#视频读取路径,此处注意与labels对应的视频相同
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置比例
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    vll=None
    frame_count = 0  # 已记录帧数
    output_path = "F:/new/new_test_videos/new_1.mp4"  # 输出路径
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出格式
    out = cv2.VideoWriter(output_path, fourcc, 90, (width, height))
    
    vll_past = None  # 初始化前一帧
    prev_time = time.time()


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        key = cv2.waitKey(1)
        if key == 113:  # 按 q 可以终止程序
            break
        if(int((frame_count-20)/step)>0 and int((frame_count-30)/step) < len(predicted_labels) ):
            current_label = predicted_labels[int((frame_count-30)/step)]
            cv2.putText(frame, f"LABEL: {current_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Camera Feed', frame)
    cap.release()

        


if __name__ == "__main__":
    
    i=int(input("请输入程序选项："))
    if(i==0):
        analyze_frames()
        save_data("cluster","tot_features2")
    elif(i==1):
        load_data("cluster","tot_features2")
    predicted_labels = cluster_plot(tot_features, 2)
    label_video(predicted_labels)