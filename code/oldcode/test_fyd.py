import cv2
import sleap
import time
import numpy as np
import tensorflow as tf
batch_size=16
predictor = sleap.load_model([r"F:\guangyichuan\models\240727_145201.centroid.n=38.zip", r"F:\guangyichuan\models\240727_150142.centered_instance.n=38.zip"], batch_size)  # 采用的 sleap 模型

prev_time = time.time()  # 记录上一帧的时间
width = 1280
height = 1024
cap = cv2.VideoCapture(r"C:\0702Sweet 5+5+5-1.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置比例
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cv2.setUseOptimized(True)
frames = np.array([])
video_size = 120 #想要预先跑的帧数
j = 0
while (j<video_size):
    ret, frame = cap.read()
    if ret:
        j+=1
        ex_frame = np.expand_dims(frame, axis=0)
        if(frames.size == 0):
            frames = ex_frame
        else:
            frames = np.concatenate((frames, ex_frame), axis=0)
    else:
        break
frame_predictions = predictor.inference_model.predict(frames)
print('num of frames in the video:',video_size) 
time_predicting = time.time() - prev_time
print('time:',time_predicting)
for i in range(video_size):
    vl = frame_predictions['instance_peaks']
    
    cv2.imshow('Camera Feed', frames[i,:,:,:])
    current_time = time.time()
    #fps = 1.0 / (current_time - prev_time)
    #cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    prev_time = current_time    
'''
while True:
    ret, frame = cap.read()
    if ret:
        if(i == batch_size):
            i=0
            

            
            frame_predictions = predictor.inference_model.predict_on_batch(frames) 
            vl = frame_predictions['instance_peaks']
            vll = vl.numpy()
            print(vll[0].shape)
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) * batch_size
            print(fps)
            prev_time = current_time
            
            frames = np.array([])
        else:
            i +=1
        
            cv2.imshow('Camera Feed', frame)
            ex_frame = np.expand_dims(frame, axis=0)
            if(frames.size == 0):
                frames = ex_frame
            else:
                frames = np.concatenate((frames, ex_frame), axis=0)
                
            
            
            
            
            
            
            

    else:
        break
    key = cv2.waitKey(1)
    #cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #print(fps)
        #cv2.imshow('Camera Feed', frame)
'''