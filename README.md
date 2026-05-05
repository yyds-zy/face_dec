### Setup with Anaconda
**Step 1.** Create Conda environment and install pytorch.
```shell
conda create -n face_env python=3.10
conda activate face_env
```
**Step 2.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).<br>
The code was tested using torch 1.11.0+cu113 and torchvision==0.12.0 
```shell
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3.** Install UNIONPROJECT
```shell
cd UNIONPROJECT
pip install -r requirements.txt
```
**Step 4.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip install cython
pip install third_part/cocoapi/PythonAPI
```

Step 5. Others
```shell
# Cython-bbox
pip install cython_bbox

# faiss
pip install faiss-gpu

# ultralytics
pip install ultralytics==8.3.127

# timm
pip install timm

# face_recognition
pip install cmake
pip install dlib
pip install face_recognition 
conda install -c conda-forge libstdcxx-ng

# for facial attribution det
pip install onnxruntime-gpu
pip install scikit-image
pip install transformers
```

<!-- for face tracking -->

```shell
cd <UNIONPROJECT_dir>

# For yolo11s infer tracker face【ReID model：faceReID_FR / vit_8_112】(可选择添加 --with_pose)
python main.py --weights weights/detector/model_11n.pt --source test/MOT17-08.mp4 --agnostic-nms --view-img --classes 0 --conf-thres 0.5 --img-size 640 --track_buffer 150 --track_high_thresh 0.7 --min_box_area 400 --proximity_thresh 0.95 --with-reid --view_img
```

<!-- 预训练模型权重保存位置 -->
model/face_attribution/attr_race_gender.py: weights/attribution_DeepFace/gender_model.onnx 和 weights/attribution_DeepFace/race_model.onnx
model/face_attribution/appearence.py: weights/SwinFace/checkpoint_step_79999_gpu_0.pt

<!-- for face detection -->

```shell
from ultralytics import YOLO
img = Image.open("/home/lenovo/New/UnionProject/test/bus.jpg")
detector = YOLOFace(weight_path='weights/detector/model_11n.pt')
results = detector.predict(img)
results[0].show()

# 结果组成
cls_ids = results_face[0].boxes.cls.cpu().numpy()
cls_conf = results_face[0].boxes.conf.cpu().numpy()
# x1y1x2y2转xywh
xyxy = results_face[0].boxes.data[..., :4].cpu().numpy()
bbox_xywh = np.stack([
    (xyxy[:, 0] + xyxy[:, 2]) / 2,
    (xyxy[:, 1] + xyxy[:, 3]) / 2,
    xyxy[:, 2] - xyxy[:, 0],
    xyxy[:, 3] - xyxy[:, 1]
], axis=1)

```

<!-- 整体环境 -->
```
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
absl-py                   2.3.1                    pypi_0    pypi
beautifulsoup4            4.13.4                   pypi_0    pypi
bzip2                     1.0.8                h5eee18b_6  
ca-certificates           2025.7.9             hbd8a1cb_0    conda-forge
certifi                   2025.7.9                 pypi_0    pypi
charset-normalizer        3.4.2                    pypi_0    pypi
click                     8.2.1                    pypi_0    pypi
cmake                     4.0.3                    pypi_0    pypi
coloredlogs               15.0.1                   pypi_0    pypi
contourpy                 1.3.2                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
cython                    3.1.2                    pypi_0    pypi
cython-bbox               0.1.5                    pypi_0    pypi
dlib                      20.0.0                   pypi_0    pypi
easydict                  1.13                     pypi_0    pypi
expat                     2.7.1                h6a678d5_0  
face-recognition          1.3.0                    pypi_0    pypi
face-recognition-models   0.3.0                    pypi_0    pypi
faiss-gpu                 1.7.2                    pypi_0    pypi
filelock                  3.18.0                   pypi_0    pypi
filterpy                  1.4.5                    pypi_0    pypi
flatbuffers               25.2.10                  pypi_0    pypi
fonttools                 4.58.5                   pypi_0    pypi
fsspec                    2025.5.1                 pypi_0    pypi
gdown                     5.2.0                    pypi_0    pypi
grpcio                    1.73.1                   pypi_0    pypi
h5py                      3.14.0                   pypi_0    pypi
hf-xet                    1.1.5                    pypi_0    pypi
huggingface-hub           0.33.2                   pypi_0    pypi
humanfriendly             10.0                     pypi_0    pypi
idna                      3.10                     pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
joblib                    1.5.1                    pypi_0    pypi
kiwisolver                1.4.8                    pypi_0    pypi
lap                       0.5.12                   pypi_0    pypi
ld_impl_linux-64          2.40                 h12ee557_0  
libffi                    3.4.4                h6a678d5_1  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libstdcxx-ng              13.2.0               hc0a3c3a_7    conda-forge
libuuid                   1.41.5               h5eee18b_0  
libxcb                    1.17.0               h9b100fa_0  
loguru                    0.7.3                    pypi_0    pypi
markdown                  3.8.2                    pypi_0    pypi
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                3.0.2                    pypi_0    pypi
matplotlib                3.10.3                   pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
motmetrics                1.4.0                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
networkx                  3.4.2                    pypi_0    pypi
ninja                     1.11.1.4                 pypi_0    pypi
numpy                     1.23.5                   pypi_0    pypi
nvidia-cublas-cu12        12.1.3.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.1.105                 pypi_0    pypi
nvidia-cudnn-cu12         8.9.2.26                 pypi_0    pypi
nvidia-cufft-cu12         11.0.2.54                pypi_0    pypi
nvidia-curand-cu12        10.3.2.106               pypi_0    pypi
nvidia-cusolver-cu12      11.4.5.107               pypi_0    pypi
nvidia-cusparse-cu12      12.1.0.106               pypi_0    pypi
nvidia-nccl-cu12          2.20.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.9.86                  pypi_0    pypi
nvidia-nvtx-cu12          12.1.105                 pypi_0    pypi
onnx                      1.18.0                   pypi_0    pypi
onnx-simplifier           0.4.36                   pypi_0    pypi
onnxoptimizer             0.3.13                   pypi_0    pypi
onnxruntime               1.22.0                   pypi_0    pypi
opencv-python             4.11.0.86                pypi_0    pypi
openssl                   3.0.16               h5eee18b_0  
packaging                 25.0                     pypi_0    pypi
pandas                    2.3.1                    pypi_0    pypi
pillow                    11.3.0                   pypi_0    pypi
pip                       25.1               pyhc872135_2  
prettytable               3.16.0                   pypi_0    pypi
protobuf                  6.31.1                   pypi_0    pypi
psutil                    7.0.0                    pypi_0    pypi
pthread-stubs             0.3                  h0ce48e5_1  
py-cpuinfo                9.0.0                    pypi_0    pypi
pycocotools               2.0                      pypi_0    pypi
pygments                  2.19.2                   pypi_0    pypi
pyparsing                 3.2.3                    pypi_0    pypi
pysocks                   1.7.1                    pypi_0    pypi
python                    3.10.18              h1a3bd86_0  
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2025.2                   pypi_0    pypi
pyyaml                    6.0.2                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
requests                  2.32.4                   pypi_0    pypi
rich                      14.0.0                   pypi_0    pypi
safetensors               0.5.3                    pypi_0    pypi
scikit-learn              1.6.1                    pypi_0    pypi
scipy                     1.15.3                   pypi_0    pypi
seaborn                   0.13.2                   pypi_0    pypi
setuptools                78.1.1          py310h06a4308_0  
six                       1.17.0                   pypi_0    pypi
soupsieve                 2.7                      pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
sympy                     1.14.0                   pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
tensorboard               2.19.0                   pypi_0    pypi
tensorboard-data-server   0.7.2                    pypi_0    pypi
termcolor                 3.1.0                    pypi_0    pypi
thop                      0.1.1-2209072238          pypi_0    pypi
threadpoolctl             3.6.0                    pypi_0    pypi
timm                      1.0.17                   pypi_0    pypi
tk                        8.6.14               h993c535_1  
torch                     2.3.0+cu121              pypi_0    pypi
torchvision               0.18.0+cu121             pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
triton                    2.3.0                    pypi_0    pypi
typing-extensions         4.14.1                   pypi_0    pypi
tzdata                    2025.2                   pypi_0    pypi
ultralytics               8.3.127                  pypi_0    pypi
ultralytics-thop          2.0.14                   pypi_0    pypi
urllib3                   2.5.0                    pypi_0    pypi
wcwidth                   0.2.13                   pypi_0    pypi
werkzeug                  3.1.3                    pypi_0    pypi
wheel                     0.45.1          py310h06a4308_0  
xmltodict                 0.14.2                   pypi_0    pypi
xorg-libx11               1.8.12               h9b100fa_1  
xorg-libxau               1.0.12               h9b100fa_0  
xorg-libxdmcp             1.1.5                h9b100fa_0  
xorg-xorgproto            2024.1               h5eee18b_1  
xz                        5.6.4                h5eee18b_1  
yacs                      0.1.8                    pypi_0    pypi
zlib                      1.2.13               h5eee18b_1 
```
