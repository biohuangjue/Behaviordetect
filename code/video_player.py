import cv2
import time
import threading

# 创建一个 VideoCapture 对象，参数为视频文件路径
cap = cv2.VideoCapture(r'"E:\guangyichuan\videos\dataset_try_mice\WIN_20240909_20_12_09_Pro.mp4"')

prev_time = 0
fps = 0

def read_video():
    global frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

frame = None
video_thread = threading.Thread(target=read_video)
video_thread.start()

while True:
    if frame is not None:
        # 计算帧率
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 在画面上显示帧率
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示视频帧
        cv2.imshow('Video Player', frame)

    # 等待按键事件，如果按下 'q' 键，退出循环
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# 等待视频读取线程结束
video_thread.join()

# 释放 VideoCapture 对象和关闭所有窗口
cap.release()
cv2.destroyAllWindows()