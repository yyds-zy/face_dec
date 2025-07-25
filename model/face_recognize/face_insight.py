from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import os
import warnings

class FaceEmbedderInsightFace:
    def __init__(self, det_size=(640, 640), args=None):
        """
        使用 InsightFace 提取人脸特征
        :param det_size: 人脸检测图像大小
        :param model_name: 模型名称（如 buffalo_l）
        :param device: 设备 (cuda / cpu)
        """
        self.out_dim = 512
        device = args.device

        # 原始路径
        pretrain_model_dir = args.pretrained_model_dir  # 'weights/ReID/insightface/buffalo_l'

        # 拆分出 model_name 和 root
        model_name = os.path.basename(pretrain_model_dir)
        root = os.path.dirname(pretrain_model_dir)

        # 检查 root 是否以 insightface 结尾
        if os.path.basename(root) != 'insightface':
            warnings.warn(
                f"⚠️ 路径 '{root}' 并非以 'insightface' 结尾，将强制重设为默认路径。",
                stacklevel=2
            )
            root = 'weights/ReID/insightface'
            model_name = 'buffalo_l' 

        self.app = FaceAnalysis(name=model_name, 
                                allowed_modules=['detection', 'recognition'],
                                root=root, 
                                providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=det_size)
        self.device = device

    def preprocess(self, img: Image.Image) -> np.ndarray:
        return np.array(img.convert("RGB"))

    def extract(self, img: Image.Image, return_boxes: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        自动人脸检测 + 特征提取
        :return: 特征向量 [N, 512] 和人脸框 [x1, y1, x2, y2]
        """
        img_np = self.preprocess(img)
        faces = self.app.get(img_np)

        # faces = [f for f in faces if f.det_score >= 0.8]

        if not faces:
            return np.zeros((0, 512)), []

        embeddings = np.array([f.normed_embedding for f in faces])
        dets = [tuple(map(int, f.bbox)) + (float(f.det_score),) for f in faces] if return_boxes else []

        return embeddings, np.array(dets, dtype=np.float32)

if __name__ == "__main__":
    from PIL import Image

    embedder = FaceEmbedderInsightFace()
    img = Image.open("/home/lenovo/New/UnionProject/test/bus.jpg")

    # 自动检测
    feats, boxes = embedder.extract(img, return_boxes=True)
    print("检测到人脸数:", len(feats))

    # 指定框提取
    feats2 = embedder.extract_with_locations(img, boxes)
    print("指定框提取特征 shape:", feats2.shape)