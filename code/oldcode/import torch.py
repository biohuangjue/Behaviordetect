import cv2
import sleap
import time
import numpy as np
import tensorflow as tf

predictor = sleap.load_model([r"F:\guangyichuan\models\240727_145201.centroid.n=38.zip", r"F:\guangyichuan\models\240727_150142.centered_instance.n=38.zip"], batch_size=16)  # 采用的 sleap 模型

prev_time = time.time()  # 记录上一帧的时间
width = 1280
height = 1024
cap = cv2.VideoCapture("F:/guangyichuan/videos/gr5a2.7.3.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置比例
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cv2.setUseOptimized(True)
ret,frame = cap.read()
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count = frame_count + 1
    if ret:
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        #cv2.imshow('Camera Feed', frame)
        
        previous=time.time()
        if(frame_count % 1 == 0):
            frame_predictions = predictor.inference_model.predict_on_batch(np.expand_dims(frame, axis=0))

        current=time.time()
        print("huamian:",current-previous)
        print(fps)

    else:
        break
    key = cv2.waitKey(1)