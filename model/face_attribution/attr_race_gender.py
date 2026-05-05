import numpy as np
import cv2
import os
from PIL import Image
from typing import Union, List, Tuple
import onnxruntime as ort
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

class FaceAttr:
    def __init__(self, model_name='Age', device: str = "cuda", args=None):
        """
        初始化模型。
        """
        self.model_name = model_name
        self.input_resize_size=(224,224)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.gender_session = ort.InferenceSession("weights/attribution_DeepFace/gender_model.onnx", providers=providers)
        self.race_session = ort.InferenceSession("weights/attribution_DeepFace/race_model.onnx", providers=providers)


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
            ], axis=0).transpose(1, 0, 2)  # shape: (n, 5, 2)
        return bboxes, kpss
    
    def extract_with_location(self, img: Image.Image, face_info: dict = None, align_face: bool = False) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        注意:
        这里输入的原始图像, 不是剪切后的patch(因为脸部的五官点采用的是原图的坐标)
        输入图像通道为bgr(cv2.imread默认格式), 归一化图像(除255)
        """
        bboxes, kpss = self.from_dict(face_info)
        preprocessed_images = []
        for i in range(bboxes.shape[0]):
            if align_face:
                assert kpss is not None, "kpss must be provided for face alignment"
                # 对人脸进行对齐
                aimg = norm_crop(img, landmark=kpss[i], image_size=224)
            else:
                x1, y1, x2, y2 = [max(0, int(coord)) for coord in bboxes[i][:-1]]
                aimg = img[y1:y2, x1:x2]
                aimg = cv2.resize(aimg, self.input_resize_size)
            preprocessed_images.append(aimg)
            
        preprocessed_images = np.array(preprocessed_images)
        gender_output = self.gender_session.run(None, {self.gender_session.get_inputs()[0].name: preprocessed_images})
        race_output = self.race_session.run(None, {self.race_session.get_inputs()[0].name: preprocessed_images})

        face_info['gender'] = gender_output[0]
        face_info['race'] = race_output[0]

        return face_info


if __name__ == '__main__':
    from PIL import Image
    import cv2
    import sys
    

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.append(project_root)

    # from model.detector.insight_det import Insight_FaceDet

    # detector = Insight_FaceDet()
    model = FaceAttr()

    img_path = "test/people_face.jpg"
    img = cv2.imread(img_path)

    # result = detector.predict(img)
    result = {"bboxes": np.array([[0, 0, 224, 224, 1]])}  # 模拟人脸检测结果
    img = img.astype(np.float32) / 255.0

    feat = model.extract_with_location(img, result)

    for key, value in feat.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            print(f"{key}: list of arrays, first shape={value[0].shape}")
        else:
            print(f"{key}: type={type(value)}")

