import cv2
import sleap
import numpy as np
import pickle
import time
import ukf_module
import os
import tkinter as tk
from tkinter import filedialog

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

label_state = 0
data = []

# 加载 sleap 模型
predictor = sleap.load_model([r"E:\guangyichuan\models\240703_172214.centroid.n=86.zip", r"E:\guangyichuan\models\240703_174034.centered_instance.n=86.zip"], batch_size=10)

def update_data(a):
    global data
    data.append(a)

def save_data(video_folder, video_name):
    data_folder = os.path.join(video_folder, "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    data_file_path = os.path.join(data_folder, f"{video_name}_data.pkl")
    try:
        with open(data_file_path, 'wb') as f3:
            pickle.dump(data, f3)
    except Exception as e:
        print(f"Error saving data: {e}")

def load_data(folder_name, video_name):
    global data
    data_file_path = os.path.join(folder_name, "data", f"{video_name}_data.pkl")
    if os.path.exists(data_file_path):
        try:
            with open(data_file_path, 'rb') as f3:
                data = pickle.load(f3)
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        data = []
        print("File not found.")

def draw_point(v, frame, a, b, c):
    global point_data
    if v.shape[1] == 1:
        point_data = v[0][0]
    if point_data is not None:
        for point in point_data:
            x, y = point
            x = min(max(x, 0), frame.shape[1] - 1)
            y = min(max(y, 0), frame.shape[0] - 1)
            cv2.circle(frame, (int(x), int(y)), 5, (a, b, c), -1)

# 初始化 UKF
ukf = [ukf_module.UnscentedKalmanFilter(np.array([0, 0]), np.eye(2), np.array([[0.1, 0], [0, 0.1]]), np.array([[0.01, 0], [0, 0.01]])) for _ in range(11)]

def analyze_video(video_path, folder_name):
    width = 1280
    height = 1024
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 90)
    vll = None
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(video_path)
    load_data(folder_name, video_name)
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            progress_percent = frame_count / total_frames * 100
            print(f"Analyzing video: {video_name}, Progress: {progress_percent:.2f}%", end='\r')
            cv2.putText(frame, f"Progress: {progress_percent:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Data Length: {len(data)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if frame_count % 5 == 0:
                frame_predictions = predictor.inference_model.predict_on_batch(np.expand_dims(frame, axis=0))
                vl = frame_predictions['instance_peaks']
                vll = vl.numpy()
                vll = np.nan_to_num(vll, nan=0)
                vll_past = None

                if vll.shape[1] == 1 and vll_past is None:
                    vll_past = vll.copy()
                if vll.shape[1] == 1:
                    if frame_count >= 0:
                        update_data(vll[0][0])
                    if vll_past.shape[1] == 1:
                        control_input = np.array([0.1, 0])
                        for i in range(11):
                            ukf[i].predict(control_input)
                            if vll[0][0][i][0]!= 0 and vll[0][0][i][1]!= 0:
                                ukf[i].update(np.array([vll[0][0][i][0], vll[0][0][i][1]]))
                            if vll[0][0][i][0] == 0:
                                vll[0][0][i][0], vll[0][0][i][1] = ukf[i].state[0], ukf[i].state[1]

            key = cv2.waitKey(1)
            if key == 113:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass

    cap.release()
    save_data(folder_name, video_name)

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        video_files = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.endswith('.mp4') or f.endswith('.avi')]
        for video_path in video_files:
            analyze_video(video_path, folder_selected)

if __name__ == '__main__':
    select_folder()