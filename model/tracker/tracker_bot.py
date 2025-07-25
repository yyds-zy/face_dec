import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import copy
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import matching
from gmc import GMC
from basetrack import BaseTrack, TrackState
from kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, update_reid=True):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None and update_reid:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, update_reid=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None and update_reid:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args, model_name=args.ReID_emb_name)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        self.init_inter_container()  # 初始化交互容器

    def init_inter_container(self):
        """初始化中间变量"""
        self.inter_container = {}

    def inter_vis_res(self, frame_count, **kwargs):
        # 最新输入一帧时先初始化
        if frame_count not in self.inter_container:
            self.inter_container[frame_count] = {}

        # 根据不同的输入更新中间变量
        for key, value in kwargs.items():
            self.inter_container[frame_count][key] = value

    def process_results(self, output_results):
        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Expand boxes: left/right by 0.2*w, top/bottom by 0.2*h
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            bboxes[:, 0] = bboxes[:, 0] - 0.2 * w
            bboxes[:, 1] = bboxes[:, 1] - 0.2 * h
            bboxes[:, 2] = bboxes[:, 2] + 0.2 * w
            bboxes[:, 3] = bboxes[:, 3] + 0.2 * h

            # if output_results.shape[1] == 5:
            #     scores = output_results[:, 4]
            #     bboxes = output_results[:, :4]
            #     classes = output_results[:, -1]
            # else:
            #     scores = output_results[:, 4] * output_results[:, 5]
            #     bboxes = output_results[:, :4]  # x1y1x2y2
            #     classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []
        return bboxes, scores, classes, dets, scores_keep, classes_keep


    def update(self, output_results, img):
        if self.frame_id == 887:
            print(1)
        
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []



        '''Extract embeddings '''
        if self.args.with_reid:

            if self.encoder.reid_input == 'ori_image':
                features_keep, det_results = self.encoder.inference(img)
                bboxes, scores, classes, dets, scores_keep, classes_keep = self.process_results(det_results)
            else:
                bboxes, scores, classes, dets, scores_keep, classes_keep = self.process_results(output_results)
                features_keep = self.encoder.inference(img, detections=dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # ********* 6_20: 删除无效检测框 *********
        # # 删除无效检测框：检查检测框是否在图像边缘且有效面积占比小且长宽比大于5或小于0.2
        # if len(detections) > 0 and img is not None:
        #     img_h, img_w = img.shape[:2]
        #     valid_detections = []
        #     for i, det in enumerate(detections):
        #         tlbr = det.tlbr
        #         x1, y1, x2, y2 = tlbr
        #         x1_clip = np.clip(x1, 0, img_w)
        #         y1_clip = np.clip(y1, 0, img_h)
        #         x2_clip = np.clip(x2, 0, img_w)
        #         y2_clip = np.clip(y2, 0, img_h)
        #         box_area = max(0, x2 - x1) * max(0, y2 - y1)
        #         valid_area = max(0, x2_clip - x1_clip) * max(0, y2_clip - y1_clip)
        #         ratio = (x2 - x1) / (y2 - y1 + 1e-6)
        #         is_valid = not (box_area > 0 and valid_area / box_area < 0.5 and (ratio > 5 or ratio < 0.2))
        #         valid_detections.append(is_valid)

        #     valid_mask_det = np.array(valid_detections, dtype=bool)
        #     detections = [det for i, det in enumerate(detections) if valid_mask_det[i]]
        # **************************************


        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)
        
        '''*** 记录匹配前所有的 历史轨迹 ***'''
        self.inter_vis_res(self.frame_id, tracked_stracks=copy.deepcopy(tracked_stracks), lost_stracks=copy.deepcopy(self.lost_stracks), removed_stracks=copy.deepcopy(self.removed_stracks))

        '''*** 记录第一次参与匹配的 历史轨迹 和 检测对象 ***'''
        self.inter_vis_res(self.frame_id, first_track=strack_pool.copy(), first_detection=detections.copy())

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            # ########## 6_20: 如果一个detection同时和多个track目标很像，先不匹配它（视为detect得到的内容质量很差，当作没有detection到）
            # if emb_dists.shape[1] != 0:
            #     mask_det = np.ones(emb_dists.shape[1], dtype=bool)
            #     mask_det_cur_feature =  np.ones(emb_dists.shape[1], dtype=bool)
            #     for i_det in range(emb_dists.shape[1]):
            #         if np.sum(emb_dists[:, i_det] < 0.25) > 1:
            #             mask_det[i_det] = False
            #             # detections[i_det].curr_feat=None
            #         elif np.any(np.triu((emb_dists[:, i_det][:, None] < 0.5) & (emb_dists[:, i_det][None, :] < 0.5) & 
            #                             (np.abs(emb_dists[:, i_det][:, None] - emb_dists[:, i_det][None, :]) < 0.1), k=1)):
            #             mask_det[i_det] = False
            #             # detections[i_det].curr_feat=None
            #         # 当在 检测置信度低 的时候，不更新外观特征
            #         if detections[i_det].score < 0.92:
            #             mask_det_cur_feature[i]=False
            #             print(str(i_det) + ": not renew appearance")
            #             emb_dists[:, i_det] += 1-((1 - emb_dists[:, i_det]) * detections[i_det].score)

            #     '''Method_1: 只保留满足条件的检测目标'''
            #     # emb_dists = emb_dists[:, mask_det]
            #     # detections = [detections[i] for i, det in enumerate(detections) if mask_det[i]]
            #     # ious_dists = ious_dists[:, mask_det]
            #     '''Method_2: re_ID特征不用'''
            #     emb_dists[:, mask_det==False] = 1.0

            # ########## 7_14: 如果一个detection和历史目标匹配上了，但是embedding差距较大，匹配它但不更新特征
            if emb_dists.shape[1] != 0:
                mask_det_cur_feature =  np.ones(emb_dists.shape[1], dtype=bool)
                for i_det in range(emb_dists.shape[1]):
                    # 当在 检测置信度低 的时候，不更新外观特征
                    if emb_dists.size and emb_dists[:, i_det].min() > self.args.similarity_thresh:
                        mask_det_cur_feature[i_det]=False

            # ########## 
            # dists = np.minimum(ious_dists, emb_dists)
            dists = emb_dists

            # # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections, only_position=False)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        '''*** 记录第一次匹配结果 ***'''
        self.inter_vis_res(self.frame_id, first_match_matrix=dists.copy())
        self.inter_vis_res(self.frame_id, first_match_matrix_ious=ious_dists.copy())
        self.inter_vis_res(self.frame_id, first_match_matrix_emb=emb_dists.copy())

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # ************** 6_20：额外的外观更新约束
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, update_reid=mask_det_cur_feature[idet])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, update_reid=mask_det_cur_feature[idet])
                refind_stracks.append(track)
        
        # ********************************************************

        # ************** 原来内容
        # for itracked, idet in matches:
        #     track = strack_pool[itracked]
        #     det = detections[idet]
        #     if track.state == TrackState.Tracked:
        #         track.update(detections[idet], self.frame_id)
        #         activated_starcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_stracks.append(track)

                
        # *** 从原先的tracked_stracks 和 self.lost_stracks 得到 activated_tracks 和 refined_stracks *** unconfirmed为not activated 的只作了光流的视角变化 ***
        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        '''*** 记录第二次参与匹配的 历史轨迹 和 检测对象 ***'''
        self.inter_vis_res(self.frame_id, second_track=r_tracked_stracks.copy(), second_detection=detections_second.copy())

        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        '''*** 记录第二次匹配结果 ***'''
        self.inter_vis_res(self.frame_id, second_match_matrix=dists.copy(), second_match=matches.copy())

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        '''*** 记录未确认检测对象的匹配结果 ***'''
        self.inter_vis_res(self.frame_id, unconfirm_match_matrix_ious=ious_dists.copy(), unconfirm_match_matrix_emb=emb_dists.copy(), 
                           unconfirm_match_matrix=dists.copy(), unconfirm_matches=matches.copy())
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]

        return output_stracks, self.inter_container[self.frame_id]


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
