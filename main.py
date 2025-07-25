import sys
import cv2
import torch
import numpy as np
from numpy import random
import time
from tqdm import tqdm

import argparse
import os
import os.path as osp

from loguru import logger
from ultis.pose_color import kpts

# 添加项目根目录到系统路径
sys.path.append('.')

from model.build import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test/video_test.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')
    parser.add_argument('--view_img', action='store_true', help='view_img')

    # detector
    parser.add_argument('--img_size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.7, help='IOU threshold for NMS')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=40000, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")
    parser.add_argument("--similarity_thresh", default=0.4, type=float, help="renew embedding thresh")
    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    
    parser.add_argument("--ReID-emb-name", dest="ReID_emb_name", default="FaceEmbedderDirect",
                        type=str, help="reid model name")
    parser.add_argument("--pretrained-model-dir", dest="pretrained_model_dir", default="weights/ReID/face_ReID_vit",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default="weights/ReID/face_ReID_vit/model.safetensors",
                        type=str, help="reid config file path")
    
    parser.add_argument("--ReID_input_size", default=640, type=int, help="only patch-input ReID model need")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    
    # pose estimation
    parser.add_argument('--with_pose', default=False, action='store_true', help='hide labels')
    

    return parser.parse_args()



class VideoWrapper:
    def __init__(self, args):
        self.args = args

        self.detector = build_detector(args)
        self.tracker = build_tracker(args, frame_rate=30)
        if args.with_pose:
            self.pose_generate = build_pose_generator(args)
        # self.mesh_generate = build_mesh_generater()

        # 用于计时，单位为秒
        self.det_infer_times = []
        self.reid_infer_times = []
        self.total_times = []
        self.detection_times = []
        self.tracking_times = []

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]
        self.save_img = not args.nosave and not args.source.endswith('.txt')  # save inference images
        
        # 视频保存相关属性
        self.video_writer = None
        self.save_path = None

        # 中间结果可视化相关
        self.all_trackes = {}

    def plot_one_box_bottom(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 6)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            # c2 = c1[0] + t_size[0], c1[1] + t_size[1] - 3
            c2 = c1[0] +  int(t_size[0]/2), c1[1] + int(t_size[1]/2) - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    def process_frame(self, frame_ori, frame_count, estimate_pose=False):
        # 拷贝一份frame用于后续修改，避免影响原始frame
        frame = frame_ori.copy()
        # 初始化视频写入器（仅在第一帧调用）
        if frame_count == 1 and not self.args.nosave:
            self._init_video_writer(frame)
        
        # 将BGR转换为RGB
        # im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 执行目标检测
        det_start = time.time()

        # 检测
        results_face = self.detector.predict(frame)
        if estimate_pose:
            results_pose = self.pose_generate.predict(frame, conf=0.3)           

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
        
        p_det_p_time = time.time() - det_start

        self.detection_times.append(p_det_p_time)

        # 仅保留person类别（cls_id=0）
        mask = cls_ids == 0
        bbox_xywh = bbox_xywh[mask]
        # bbox_xywh[:, 2:] *= 1.2  # 扩展边界框
        cls_conf = cls_conf[mask]
        cls_ids = cls_ids[mask]

        # 添加面积过滤
        if len(bbox_xywh) > 0:
            # 计算检测框面积 (w*h)
            areas = bbox_xywh[:, 2] * bbox_xywh[:, 3]
            
            # 过滤面积小于阈值的检测框
            area_mask = areas >= self.args.min_box_area
            bbox_xywh = bbox_xywh[area_mask]
            cls_conf = cls_conf[area_mask]
            cls_ids = cls_ids[area_mask]

        # 准备BoT-SORT输入
        detections = []
        if len(bbox_xywh) > 0:
            # 转换边界框格式: xywh -> x1y1x2y2
            bbox_x1y1x2y2 = bbox_xywh.copy()
            bbox_x1y1x2y2[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2  # x1 = x - w/2
            bbox_x1y1x2y2[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2  # y1 = y - h/2
            bbox_x1y1x2y2[:, 2] = bbox_xywh[:, 0] + bbox_xywh[:, 2]/2  # x2 = x + w/2
            bbox_x1y1x2y2[:, 3] = bbox_xywh[:, 1] + bbox_xywh[:, 3]/2  # y2 = y + h/2
            
            # 组装BoT-SORT所需的detections格式
            detections = np.hstack((
                bbox_x1y1x2y2,
                cls_conf.reshape(-1, 1)
                # cls_ids.reshape(-1, 1)
            ))

        # 执行跟踪
        online_targets, current_inter_res = self.tracker.update(detections, frame)

        # 绘制检测框
        online_tlwhs = []
        online_ids = []
        online_scores = []
        
        if estimate_pose:
            for i, k in enumerate(reversed(results_pose[0].keypoints.data)):
                kpts(
                    frame,
                    k,
                    frame.shape[:2],
                )

        results_return = []
        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id

            if tlwh[2] * tlwh[3] > self.args.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                
                # save results
                results_return.append(
                    f"{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )

                label = f'{tid}, person'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_y = int(tlbr[3]) + label_size[1] + 5  # Position label below the box
                label_x = int(tlbr[0])
                # plot_one_box(tlbr, frame, color=self.colors[int(tid) % len(self.colors)], line_thickness=2)
                self.plot_one_box_bottom(tlbr, frame, color=self.colors[int(tid) % len(self.colors)], line_thickness=2)
                cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[int(tid) % len(self.colors)], 1)

        # print(f'results:{results_return} \n')
        
        # 保存视频帧
        if not self.args.nosave and self.video_writer is not None:
            # 将frame与vis_addition拼接
            self.video_writer.write(frame)

        # return frame, results
        return frame, results_return, current_inter_res

    def _init_video_writer(self, frame):
        """初始化视频写入器"""
        # 设置保存目录
        save_dir = os.path.join(self.args.project, self.args.name, self.args.ReID_emb_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成保存文件名
        if self.args.source.isnumeric():
            save_name = "webcam_result.mp4"
        else:
            file_name = os.path.basename(self.args.source)
            save_name = f"{os.path.splitext(file_name)[0]}_result.mp4"
        
        self.save_path = os.path.join(save_dir, save_name)
        
        # 获取视频属性
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) if 'cap' in globals() else 30  # 尝试获取实际FPS，默认为30
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.save_path, fourcc, fps, (width, height))
        
        print(f"视频将保存至: {self.save_path}")

    def release_video_writer(self):
        """释放视频写入器资源"""
        if self.video_writer is not None:
            self.video_writer.release()
            if self.save_path:
                print(f"视频保存完成: {self.save_path}")
            self.video_writer = None
            self.save_path = None

if __name__ == "__main__":
    args = parse_args()
    args.ablation = False
    
    print(f"模型加载成功，设备: {args.device}, 输入尺寸: {args.img_size}")

    # 读取视频文件或摄像头
    cap = cv2.VideoCapture(args.source if args.source.isnumeric() else args.source)
    if not cap.isOpened():
        print(f"无法打开视频源: {args.source}")
        sys.exit(1)
        
    wrapper = VideoWrapper(args)
    print(f"视频FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")

    frame_count = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_time = time.time()  # 记录开始时间
    try:
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("视频读取完毕")
                    break

                frame_with_boxes, tracking_results, current_inter_res = wrapper.process_frame(frame, frame_count, estimate_pose=args.with_pose)

                # if args.view_img:
                #     # cv2.namedWindow("Tracking Result", cv2.WINDOW_NORMAL)
                #     # cv2.resizeWindow("Tracking Result", 960, 540)
                #     cv2.imshow("Tracking Result", frame_with_boxes)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # 进度条更新
                pbar.update(1)
                frame_count += 1

    finally:
        end_time = time.time()  # 记录结束时间
        elapsed = end_time - start_time
        print(f"\n视频处理完成，总耗时：{elapsed:.2f} 秒")
        
        wrapper.release_video_writer()
        cap.release()
    #     cv2.destroyAllWindows()

    # frame_count = 1
    # try:

    #     while cap.isOpened():
    #         # if frame_count == 180:
    #         #     print('1')
    #         ret, frame = cap.read()
    #         if not ret:
    #             print("视频读取完毕")
    #             break

    #         frame_with_boxes, tracking_results, current_inter_res = wrapper.process_frame(frame, frame_count, estimate_pose=args.with_pose)

    #         frame_count += 1

    # finally:
    #     # 确保资源释放
    #     wrapper.release_video_writer()
    #     cap.release()
    # #     cv2.destroyAllWindows()

    
