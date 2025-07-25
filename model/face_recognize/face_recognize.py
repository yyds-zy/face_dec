import face_recognition
import numpy as np
from PIL import Image
from typing import List, Tuple

class FaceEmbedderFR:
    def __init__(self, model: str = "small", num_jitters: int = 1, args=None):
        """
        :param model: 人脸对齐模型，"small" 或 "large"
        :param num_jitters: 抖动次数，越大越精确但越慢
        """
        self.model = model
        self.out_dim = 512
        self.num_jitters = num_jitters

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """
        PIL.Image → np.ndarray（RGB）
        """
        img = img.convert("RGB")
        return np.array(img)

    def extract(self, img: Image.Image) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        :param img: PIL.Image 格式
        :return: Tuple -> [N, 128] 特征数组 + 对应的人脸位置列表
        """
        image_np = self.preprocess(img)

        # Step 1: 检测人脸框
        face_locations = face_recognition.face_locations(image_np, model="hog")  # or "cnn"

        # Step 2: 提取特征
        encodings = face_recognition.face_encodings(
            image_np,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters,
            model=self.model
        )

        return np.array(encodings), face_locations
    
    def extract_with_locations(self, img: Image.Image, known_face_locations: List[Tuple[int, int, int, int]]) -> np.ndarray:
        image_np = self.preprocess(img)
        encodings = face_recognition.face_encodings(
            image_np,
            known_face_locations=known_face_locations,
            num_jitters=self.num_jitters,
            model=self.model
        )
        return np.array(encodings)  # shape: (num_faces, 128)

if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    # 示例图像（你可以换成真实图像）
    img = Image.open("/home/lenovo/New/UnionProject/test/bus.jpg")

    embedder = FaceEmbedderFR(model="large")

    features = embedder.extract(img)
    print("提取到的人脸数:", features.shape[0])
    if features.shape[0] > 0:
        print("第一张人脸的前5维特征向量:", features[0][:5])
