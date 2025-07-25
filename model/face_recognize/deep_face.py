from deepface import DeepFace
import numpy as np
from PIL import Image
import torch

# class FaceEmbedderDeepFace:
#     def __init__(self, model_name='VGG-Face',device: str = "cuda", args=None):
#         '''
#         model_name已将权重下载到本地的有"VGG-Face", "DeepFace", "DeepID", "GhostFaceNet", 
#         '''
#         self.input_resize_size = None
#         self.model_name = model_name
#         if model_name == 'VGG-Face':
#             self.out_dim = 4096
#         elif model_name == 'DeepFace':
#             self.out_dim = 4096
#         elif model_name == 'DeepID':
#             self.out_dim = 160
#         elif model_name == 'GhostFaceNet':
#             self.out_dim = 512
#         self.device = args.device if args!=None else 'cpu'

#     def extract(self, img) -> np.ndarray:
#         """
#         支持输入 PIL.Image、torch.Tensor 或 np.ndarray
#         返回人脸 embedding 向量
#         """
#         # 转换为 numpy
#         img_np = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#         # TODO: 好像deepface自带了有关patch相关的prepocess来提升准确程度
#         self.input_resize_size = (224, 224)  # (112, 112)

#         import tensorflow as tf
#         tf.debugging.set_log_device_placement(True)
#         # 用于 DeepFace 提特征
#         embedding_objs = DeepFace.represent(
#             img_path=img_np,
#             model_name=self.model_name,
#             detector_backend='skip',
#             enforce_detection=False
#         )

#         if not embedding_objs:
#             raise ValueError("No face embedding returned.")
        
#         if len(embedding_objs) > 1:
#             face_embrdding = np.array([embedding_objs[i][0]["embedding"] for i in range(len(embedding_objs))])
#             img_tensor = torch.from_numpy(face_embrdding)
#         else:
#             face_embrdding = np.array(embedding_objs[0]["embedding"])
#             img_tensor = torch.from_numpy(face_embrdding).unsqueeze(0)
#         return img_tensor.to(img.device)

import numpy as np
import torch
from collections import defaultdict
from deepface.modules import modeling, preprocessing
import os

class FaceEmbedderDeepFace:
    def __init__(self, model_name='VGG-Face', device: str = "cuda", args=None):
        # os.environ['LD_LIBRARY_PATH'] = '/media/lenovo/ce79d608-7210-414a-a971-bfeb5d58d04c/usr/local/cuda-11.3/targets/x86_64-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

        """
        初始化 DeepFace 模型，只支持已下载好的四个模型。
        """
        self.model_name = model_name
        self.device = device
        self.input_resize_size=(224,224)
        # 设置输出维度
        if model_name in ['VGG-Face', 'DeepFace']:
            self.out_dim = 4096
        elif model_name == 'DeepID':
            self.out_dim = 160
        elif model_name == 'GhostFaceNet':
            self.out_dim = 512
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 初始化模型
        self.model = modeling.build_model(
            task="facial_recognition",
            model_name=model_name
        )
        self.input_shape = self.model.input_shape

    def extract(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, C, H, W] 的 torch.Tensor，RGB格式，范围[0,255]
        return: [B, out_dim] 的 embedding
        """
        # 转为 numpy BGR 格式
        imgs = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[:, :, :, ::-1]  # [B, H, W, C]
        batch_images = []

        for i in range(imgs.shape[0]):
            # resize + normalize
            bgr = imgs[i]
            resized = preprocessing.resize_image(bgr, (self.input_shape[1], self.input_shape[0]))
            normed = preprocessing.normalize_input(resized, normalization="base").squeeze(0)
            batch_images.append(normed)

        batch_input = np.stack(batch_images, axis=0)
        embeddings = np.array(self.model.forward(batch_input))
        if len(batch_images)==1:
            embeddings = embeddings[None,]
        return torch.from_numpy(embeddings).to(img.device)


if __name__ == '__main__':
    from torchvision import transforms

    tensor_img = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    embedder = FaceEmbedderDeepFace(model_name='DeepFace')
    feat = embedder.extract(tensor_img)

    print("Embedding shape:", feat.shape)
    print("Embedding (first 5 dims):", feat[:5])
