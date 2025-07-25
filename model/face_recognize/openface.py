import numpy as np
from PIL import Image
from typing import List, Tuple
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms


class FaceEmbedderFacenet:
    def __init__(self, args=None):
        """
        使用 facenet-pytorch 实现人脸特征提取
        :param device: "cuda" or "cpu"
        """
        self.device = args.device
        self.out_dim = 512
        self.detector = MTCNN(keep_all=True, device=self.device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [-1, 1]
        ])

    def preprocess(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    def extract(
        self, 
        img: Image.Image, 
        return_boxes: bool = False
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        自动检测并提取人脸特征
        """
        img = self.preprocess(img)
        boxes, probs = self.detector.detect(img)

        if boxes is None:
            return np.zeros((0, 512)), []

        faces = self.detector.extract(img, boxes, save_path=None).to(self.device)

        with torch.no_grad():
            embeddings = self.embedder(faces).cpu().numpy()

        boxes_list = [tuple(map(int, box)) for box in boxes] if return_boxes else []

        return embeddings, boxes_list

    def extract_with_locations(
        self, 
        img: Image.Image, 
        known_face_locations: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        使用指定的人脸框位置提取人脸特征（不再自动检测）
        :param img: 输入图像（PIL）
        :param known_face_locations: [(top, right, bottom, left)] 格式的人脸框列表
        :return: 特征数组 [N, 512]
        """
        img_rgb = np.array(self.preprocess(img))
        faces = []

        for (top, right, bottom, left) in known_face_locations:
            face = img_rgb[top:bottom, left:right]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            face_pil = Image.fromarray(face)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            faces.append(face_tensor)

        if not faces:
            return np.zeros((0, 512))

        batch = torch.cat(faces, dim=0)

        with torch.no_grad():
            embeddings = self.embedder(batch).cpu().numpy()

        return embeddings
    
if __name__ == '__main__':
    from PIL import Image

    img = Image.open("/home/lenovo/New/UnionProject/test/bus.jpg")
    embedder = FaceEmbedderFacenet()
    features, boxes = embedder.extract(img, return_boxes=True)

    print("人脸数量:", len(features))
    if len(features):
        print("第一张人脸特征前5维:", features[0][:5])
        print("第一张人脸框:", boxes[0])