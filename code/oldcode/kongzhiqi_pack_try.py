from time import perf_counter
import numpy as np
import cv2
import sleap
import tensorflow as tf
print(cv2.__version__)
import serial
import time
import matplotlib.pyplot as plt
from datetime import datetime
from pynput import keyboard
import csv

ser = serial.Serial('COM3', 9600, timeout=1)
user = 0
jdg = 0
key_pressed=False
ve1=0
ve2=0

def on_press(key):
    global key_pressed
    if key == keyboard.KeyCode.from_char('q'):
        key_pressed = True
        
def send_command(command):
    ser.write(command.encode('utf-8'))
    ser.flush()

class RealTimeVideoAnalyzer:
    def __init__(self):
        # 初始化摄像头捕获对象
        self.cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，如果您有其他摄像头设备，可以更改这个数字
        width = 1280
        height = 1024
        print("当前宽度:", width)
        print("当前高度:", height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.csv_file ="C:\Users\Public\Desktop\lll.csv"  # CSV 文件路径
            
        
        
        # 定义视频写入器，并指定输出目录和 MP4 格式
        output_path = "F:/new/new_test_videos/new_1.mp4" # 更改为您想要的目录和文件名
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码格式
        self.out = cv2.VideoWriter(output_path, fourcc, 140.0, (width, height))  # 输出文件名、编码格式、帧率、分辨率

        # 加载预测器
        self.predictor = sleap.load_model(["F:/new/models/240703_172214.centroid.n=86.zip", "F:/new/models/240703_174034.centered_instance.n=86.zip"], batch_size=16)
        self.frame_count = 0
        self.vll_past = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])  # 用于存储 50 帧前的 vll
        self.behavior_plot = np.zeros(0)  # 用于存储行为标记的数组
        self.speed_plot = np.zeros(0)  # 正确初始化 speed_plot
        self.fly_positions = []  # 用于存储果蝇的坐标

        
        #键盘
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def analyze_frame(self): #对提取到的图像进行分析，获得vl是张量数据，vll是转换成的numpy数组
        global jdg
        global ve1
        global ve2
        ret, frame = self.cap.read()  # 读取一帧视频
        self.frame_count += 1  # 帧数加 1
        cv2.imshow('Camera Feed', frame)
        self.out.write(frame)

        if ret and self.frame_count % 50 == 0:
            # 进行推理
            t0 = perf_counter()
            frame_predictions = self.predictor.inference_model.predict_on_batch(np.expand_dims(frame, axis=0))
            dt = perf_counter() - t0
            vl = frame_predictions['instance_peaks']
            vll = vl.numpy()

            a = 1
            b = 6
            c = 5

            if vll.shape[1] == 1:
                x1, y1 = vll[0][0][a][0], vll[0][0][a][1]
                x2, y2 = vll[0][0][b][0], vll[0][0][b][1]
                x3, y3 = vll[0][0][c][0], vll[0][0][c][1]
                #dst = ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** (1 / 2)
                if self.vll_past.shape[1]==1:
                    x11, y11 = self.vll_past[0][0][a][0], self.vll_past[0][0][a][1]
                    x21, y21 = self.vll_past[0][0][b][0], self.vll_past[0][0][b][1]
                    x31, y31 = self.vll_past[0][0][c][0], self.vll_past[0][0][c][1]
                    ve=((x11 - x1) ** 2 + (y11 - y1) ** 2) ** (1 / 2)
                    self.speed_plot = np.append(self.speed_plot, np.full(50,ve))  # 存储速度值
                    self.fly_positions.append((x1, y1, self.frame_count))
                    print(ve)
                    self.save_to_csv(ve, x1, y1)  # 保存数据到 CSV
                    if y1<512 :#识别条件
                        self.behavior_plot = np.append(self.behavior_plot, np.ones(50))
                        send_command('Y')
                        #bin_index = self.frame_count // 5000
                        #self.y_counts[bin_index] += 1  # 统计 Y 输出次数
                        ve1+=ve
                    
                    else:
                        self.behavior_plot = np.append(self.behavior_plot, np.zeros(50))
                        send_command('N')
                        ve2+=ve
            if self.frame_count >= 50 and self.vll_past is not None:  # 如果有 50 帧前的数据且不为空
                # 在此处进行当前 vll 和 50 帧前的 vll_past 的比较或计算
             self.vll_past = vll.copy()  # 存储当前 vll 作为 50 帧前的数据
        
            
    def save_to_csv(self, ve, x, y):
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ve, x, y])
    
    def run(self):#运行与停止
        while True:
            self.analyze_frame()
            key = cv2.waitKey(1)
            if key_pressed:
            
            
            #print(key)
            #if key == ord('q'):  # 明确判断键值是否为 'q'
                send_command('N')
                break

        self.cap.release()  # 释放摄像头
        self.out.release()  # 释放视频写入器
        cv2.destroyAllWindows()  # 关闭所有窗口
        send_command('N')
        
         # 绘制行为和速度图像
        fig, (ax1, ax3) = plt.subplots(1, 2)  # 创建两个子图

        ax1.plot(self.behavior_plot, label='Behavior')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Behavior')
        #ax2=fig.add_subplot(121) # 在第一个位置添加子图
        #ax2.plot(self.speed_plot, color='red', label='Speed')
        #ax2.set_ylabel('Speed')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(self.speed_plot, color='red', label='Speed')
        ax1_twin.set_ylabel('Speed')
        # 绘制果蝇轨迹图
        #x_positions, y_positions = zip(*self.fly_positions)
        x_positions, y_positions, frame_counts = zip(*self.fly_positions)
        speeds = self.speed_plot[:len(self.fly_positions)]
        min_speed = np.min(speeds)
        max_speed = np.max(speeds)
        
        min_frame_count = np.min(frame_counts)
        max_frame_count = np.max(frame_counts)
        
        cmap = plt.cm.get_cmap('viridis')  # 选择颜色映射
        #norm = plt.Normalize(min_speed, max_speed)  # 归一化速度值
        norm = plt.Normalize(min_frame_count, max_frame_count)  # 归一化帧数
 
        for i in range(len(x_positions) - 1):
            ax3.plot([x_positions[i], x_positions[i + 1]], [y_positions[i], y_positions[i + 1]], color=cmap(norm(frame_counts[i])))
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        # 绘制速度对应颜色的色带
        #fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax3, label='Frame Count') 
        #fig.colorbar(sm, ax=ax3, ticks=np.linspace(min_speed, max_speed, 5))
        
        # 绘制 Y 输出次数的柱状图
        #bins = range(0, self.frame_count + 5000, 5000)
        #ax3 = fig.add_subplot(133)  # 在第三个位置添加子图
        #ax3.bar(bins[:-1], self.y_counts)
        #ax3.set_xlabel('Frame Range')
        #ax3.set_ylabel('Y Output Count')

        plt.title('Behavior, Speed')
        plt.show()
        
        plt.title('Behavior, Speed, and Fly Trajectory')
        plt.show()
       #####狠狠的改序号
if __name__ == "__main__":
    analyzer = RealTimeVideoAnalyzer()
    analyzer.run()