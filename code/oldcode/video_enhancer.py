import cv2
import numpy as np

def enhance_video_quality(frame):
    # 图像锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_frame = cv2.filter2D(frame, -1, kernel)

    # 中值滤波去噪
    denoised_frame = cv2.medianBlur(sharpened_frame, 3)

    return denoised_frame
def transfer_video(input_video_path, output_video_path):
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)

    # 获取输入视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义输出视频的编码格式和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 在这里对帧进行处理，例如调用前面的 enhance_video_quality 函数
        enhanced_frame = enhance_video_quality(frame)

        # 将处理后的帧写入输出视频
        out.write(enhanced_frame)

    # 释放资源
    cap.release()
    out.release()

# 调用函数，指定输入和输出视频路径
input_video = "F:/guangyichuan/videos/gr66c 3.7.1-1.mp4"
output_video = "F:/guangyichuan/videos/gr66c 3.7.1-1-1.mp4"
transfer_video(input_video, output_video)