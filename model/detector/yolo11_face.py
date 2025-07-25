from ultralytics import YOLO

class YOLOFace:
    def __init__(self, args, weight_path='weights/detector/model_11n.pt'):
        # 加载训练模型
        self.model = YOLO(weight_path)
        self.conf=args.conf_thres

    def predict(self, image):
        """
        image: 可为图像路径、PIL图像、NumPy数组或 OpenCV 图像。
        conf: 置信度阈值。
        """
        results = self.model.predict(image, save=False, imgsz=640, conf=self.conf, verbose=False)
        return results

# 示例使用
if __name__ == '__main__':
    # detector = YOLOFace(weight_path='weights/detector/feryolo-11x-64.pt')
    detector = YOLOFace(weight_path='weights/detector/model_11n.pt')
    # results = detector.predict('test/crowdhuman_test.jpg')

    import cv2
    cap = cv2.VideoCapture('/home/lenovo/New/UnionProject/test/MOT17-08.mp4')
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite('first_frame.jpg', frame)
        results = detector.predict(frame)
    else:
        raise RuntimeError("Failed to read the first frame from the video.")
    
    results[0].show()
