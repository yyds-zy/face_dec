from ultralytics import YOLO

class YOLOPose:
    def __init__(self, args, weight_path='weights/pose/yolov8n-pose.pt', device='cuda'):
        # 加载姿态关键点检测模型
        self.model = YOLO(weight_path)
        self.device = device
        self.model.to(device)
        self.conf = args.conf_thres

    def predict(self, image, conf=0.25):
        """
        进行人体骨骼关键点检测
        image: 支持路径、PIL、numpy等格式
        conf: 置信度阈值
        返回模型推理结果对象
        """
        results = self.model.predict(source=image, conf=conf, imgsz=640)
        return results

if __name__ == '__main__':
    detector = YOLOPose(weight_path='weights/pose/yolov8n-pose.pt')

    import cv2
    cap = cv2.VideoCapture('/home/lenovo/New/UnionProject/test/MOT17-08.mp4')
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("读取视频帧失败")

    # 预测
    results = detector.predict(frame, conf=0.3)

    # 如果想保存结果图
    results[0].show()
