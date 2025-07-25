# face_embedder.py

import torch
import timm
import torch.nn.functional as F
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import json
import os
import warnings


class FaceEmbedderDirect:
    def __init__(self, 
                 args=None,
                #  model_dir: str = "local-dir:weights/ReID/face_ReID_vit", 
                #  ckpt_path: str = "weights/ReID/face_ReID_vit/model.safetensors", 
                 model_kwargs: dict = None,
                 device: str = "cuda", 
                 channels_last=False, 
                 compile_model=False):
        self.device = args.device
        self.out_dim = 512
        pretrained_model_dir = args.pretrained_model_dir
        model_name = "local-dir:" + pretrained_model_dir
        ckpt_path = args.fast_reid_weights

        # 检查 root 是否以 insightface 结尾
        if os.path.basename(model_name) != 'face_ReID_vit':
            warnings.warn(
                f"⚠️ 路径 '{model_name}' 并非以 'face_ReID_vit' 结尾，将强制重设为默认路径。",
                stacklevel=2
            )
            model_name: str = "local-dir:weights/ReID/face_ReID_vit"
            ckpt_path: str = "weights/ReID/face_ReID_vit/model.safetensors"
            pretrained_model_dir: str = "weights/ReID/face_ReID_vit"

        # 输入形式
        with open(pretrained_model_dir + "/config.json", "r") as f:
            cfg = json.load(f)

        # 提取输入尺寸 [C, H, W]
        input_size = cfg["pretrained_cfg"]["input_size"]  # [3, 112, 112]
        _, height, width = input_size

        # 转换为 OpenCV resize 所需格式
        self.input_resize_size = (width, height)  # (112, 112)

        # 创建模型结构
        self.model = timm.create_model(model_name, pretrained=False, **(model_kwargs or {})).eval().to(self.device)

        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # 加载权重
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        self.model.load_state_dict(state_dict)

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((112, 112))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5  # Normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        x: torch.Tensor of shape (N, 3, 112, 112)
        Returns: np.ndarray of shape (N, embedding_dim)
        """
        with torch.no_grad():
            x = x.to(self.device, dtype=torch.float32)
            self.model = self.model.to(dtype=torch.float32)
            embs = self.model(x)
            embs = F.normalize(embs.float(), dim=1).half()
        return embs


if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    # 创建一个随机图像（你也可以用 Image.open("your_face.jpg")）
    img = Image.fromarray(np.uint8(np.random.rand(112, 112, 3) * 255))

    embedder = FaceEmbedderDirect(
        model_name="local-dir:weights/ReID/face_ReID_vit",  # 结构必须与训练时一致
        ckpt_path="weights/ReID/face_ReID_vit/model.safetensors"
    )

    feat = embedder.extract(img)
    print("Embedding shape:", feat.shape)
    print("Embedding (first 5 dims):", feat[:5])