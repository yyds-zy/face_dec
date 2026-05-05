from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import numpy as np
import cv2
from typing import List
from skimage import transform as trans

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


class FacialAgeDetector:
    def __init__(self, model_path: str = "weights/fairface_age_image_detection"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path).to(self.device)

    def _compute_age_mids(self, age_labels: List[str]) -> List[float]:
        mids = []
        for label in age_labels:
            if '-' in label:
                low, high = map(int, label.split('-'))
                mids.append((low + high) / 2)
            else:
                mids.append(float(label))  # 如果是数字直接处理
        return mids

    def from_dict(self, data):
        bboxes = data["bboxes"]
        kpss = None
        if all(key in data and data[key] is not None for key in ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]):
            kpss = np.stack([
                np.array(data["left_eye"]),
                np.array(data["right_eye"]),
                np.array(data["nose"]),
                np.array(data["left_mouth"]),
                np.array(data["right_mouth"])
            ], axis=0).transpose(1, 0, 2)
        return bboxes, kpss

    def age_to_category(self, index: int) -> str:
        categories = [
            "0-2",
            "3-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "more than 70"
        ]
        if 0 <= index < len(categories):
            return categories[index]
        else:
            raise ValueError("Invalid age category index")

    def extract_with_location(self, img: Image.Image, face_info: dict, align_face: bool = False) -> dict:
        img_np = np.array(img)
        bboxes, kpss = self.from_dict(face_info)

        age_distributions = []
        age_labels = []
        age_expecteds = []

        for i in range(bboxes.shape[0]):
            if align_face:
                assert kpss is not None, "kpss must be provided for face alignment"
                face_crop = norm_crop(img_np, landmark=kpss[i], image_size=224)
            else:
                x1, y1, x2, y2 = [max(0, int(coord)) for coord in bboxes[i][:-1]]
                face_crop = img_np[y1:y2, x1:x2]
                face_crop = cv2.resize(face_crop, (224, 224))

            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)   # BGR → RGB
            face_pil = Image.fromarray((face_crop_rgb * 255).astype(np.uint8))  # 归一化回 [0,255] 并转 PIL
            inputs = self.processor(images=face_pil, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()

            # 预测标签
            age_label = self.age_to_category(np.argmax(probs))
            age_distributions.append(probs)
            age_labels.append(age_label)

        face_info["age_distribution"] = age_distributions
        face_info["age_label"] = age_labels

        return face_info
    
if __name__ == '__main__':
    # 初始化情绪识别模型
    detector = FacialAgeDetector()

    # 使用模型进行预测
    img_path = "/home/lenovo/New/UnionProject/test/lfw_funneled/Andreas_Vinciguerra/Andreas_Vinciguerra_0001.jpg"
    img = cv2.imread(img_path)
    img = img.astype(np.float32) / 255.0
    info = {"bboxes": np.array([[0, 0, 224, 224, 1]])}

    age = detector.extract_with_location(img, info)

    # 输出每个年龄阶段的得分
    print(age)
