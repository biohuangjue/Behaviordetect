import cv2

class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.play_speed = 1
        self.is_playing = False
        self.labels = [None] * self.frame_count

    def display_frame(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.putText(frame, f"Frame: {self.current_frame}/{self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Video Player', frame)
            self.current_frame += 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0

    def play_video(self):
        while True:
            if self.is_playing:
                for _ in range(self.play_speed):
                    self.display_frame()
            key = cv2.waitKey(10)
            if key == ord('q'):#q停止
                break
            elif key == ord('p'):#p暂停
                self.is_playing = not self.is_playing
            elif key == ord('+'):#加速
                self.play_speed += 1
            elif key == ord('-'):#减速
                if self.play_speed > 1:
                    self.play_speed -= 1
            elif key == ord('s'):#标注
                start_frame = int(input("Enter start frame for labeling: "))
                end_frame = int(input("Enter end frame for labeling: "))
                label = input("Enter label: ")
                for frame_num in range(start_frame, end_frame + 1):
                    self.labels[frame_num] = label

        self.cap.release()
        cv2.destroyAllWindows()
        return self.labels

video_path = r"E:\5772ad98f63163e2c813102bb74d84e3.mp4"#视频的路径
player = VideoPlayer(video_path)
labels = player.play_video()
print(labels)