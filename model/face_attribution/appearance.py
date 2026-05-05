import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from PIL import Image

from model_base.swin import SwinTransformer
from model_base.subnets import *

class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size = 3
    fam_in_chans = 2112
    fam_conv_shared = False
    fam_conv_mode = "split"
    fam_channel_attention = "CBAM"
    fam_spatial_attention = None
    fam_pooling = "max"
    fam_la_num_list = [2 for _ in range(11)]
    fam_feature = "all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512


class SwinFaceAttributeExtractor:
    def __init__(self, weight_path: str='weights/SwinFace/checkpoint_step_79999_gpu_0.pt', input_resize_size: Tuple[int, int] = (112, 112), device: str = 'cuda'):
        cfg = SwinFaceCfg()
        self.input_resize_size = input_resize_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        backbone = SwinTransformer(num_classes=512)

        fam = FeatureAttentionModule(
            in_chans=cfg.fam_in_chans, kernel_size=cfg.fam_kernel_size, 
            conv_shared=cfg.fam_conv_shared, conv_mode=cfg.fam_conv_mode, 
            channel_attention=cfg.fam_channel_attention, spatial_attention=cfg.fam_spatial_attention,
            pooling=cfg.fam_pooling, la_num_list=cfg.fam_la_num_list)
        tss = TaskSpecificSubnets()
        om = OutputModule()

        self.model = ModelBox(backbone=backbone, fam=fam, tss=tss, om=om, feature=cfg.fam_feature)

        self._load_weights(weight_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, weight_path: str):
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.backbone.load_state_dict(checkpoint["state_dict_backbone"])
        self.model.fam.load_state_dict(checkpoint["state_dict_fam"])
        self.model.tss.load_state_dict(checkpoint["state_dict_tss"])
        self.model.om.load_state_dict(checkpoint["state_dict_om"])

    def from_dict(self, face_info: dict) -> Tuple[np.ndarray, np.ndarray]:
        bboxes = np.array(face_info['bboxes'])  # Nx5
        kpss = np.array(face_info['kps']) if 'kps' in face_info else None
        return bboxes, kpss

    @torch.no_grad()
    def infer_single(self, img: np.ndarray) -> Dict[str, Any]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.sub_(0.5).div_(0.5).to(self.device)

        output = self.model(img)
        results = {}
        for key, val in output.items():
            val = val[0].cpu().numpy()
            if key == 'Age':
                results[key] = float(output[key][0][0])
            elif len(val) == 2:
                prob = torch.softmax(torch.tensor(val), dim=0)
                results[key] = float(prob[1])
            elif key == 'Expression':
                prob = torch.softmax(torch.tensor(val), dim=0)
                results[key] = list(prob.cpu().numpy())
            else:
                results[key] = list(val)
        return results

    def extract_with_location(self, img: Image.Image, face_info: dict = None, align_face: bool = False) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        img_np = np.array(img)
        bboxes, kpss = self.from_dict(face_info)
        appearance_info = {}
        face_regions = []

        for i in range(bboxes.shape[0]):
            if align_face:
                from face_align import norm_crop  # 必须确保有此模块,暂时先不用
                assert kpss is not None, "kpss must be provided for face alignment"
                face_img = norm_crop(img_np, landmark=kpss[i], image_size=self.input_resize_size[0])
            else:
                x1, y1, x2, y2 = [max(0, int(coord)) for coord in bboxes[i][:-1]]
                face_img = img_np[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, self.input_resize_size)
                face_regions.append((x1, y1, x2, y2))

            result = self.infer_single(face_img)
            appearance_info[i] = result

        # 合并每个人脸的结果到 appearance_dict
        appearance_dict = {}
        for key in result:
            combine_res = []
            for i in range(len(appearance_info)):
                if key not in appearance_info[i].keys():
                    raise RuntimeError(f"Key '{key}' not found in appearance_info[{i}]")
                else:
                    combine_res.append(appearance_info[i][key])
            arr = np.array(combine_res)
            appearance_dict[key] = arr

        face_info["appearance"] = appearance_dict
        return face_info


# 示例用法
if __name__ == "__main__":
    from PIL import Image
    import json

    image_path = "/home/lenovo/New/UnionProject/test/IMDB_wiki/wiki_crop/13/226113_1963-08-21_2013.jpg"

    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255.0

    face_info = {
        'bboxes': [[0, 0, 220, 220, 0.99], [0, 0, 220, 220, 0.99]],
    }

    extractor = SwinFaceAttributeExtractor(weight_path='weights/SwinFace/checkpoint_step_79999_gpu_0.pt')
    face_info = extractor.extract_with_location(img, face_info, align_face=False)

    print("Updated face_info['appearance']:")
    print(face_info['appearance'])
