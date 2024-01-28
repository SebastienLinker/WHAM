from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
from progress.bar import Bar

from ultralytics import YOLO
import mmpose
if mmpose.__version__[0] == '0':
    from mmpose.apis import (
        inference_top_down_pose_model,
        init_pose_model,
        get_track_id,
    )
else:
    from mmpose.apis import init_model, inference_topdown, _track_by_iou


def new_get_track_id(results, results_last, next_id, min_keypoints=3, tracking_thr=0.3):
    for result in results:
        track_id, results_last, match_result = _track_by_iou(result, results_last,
                                                      tracking_thr)
        if track_id == -1:
            if np.count_nonzero(result.pred_instances.keypoints[0, :, 1]) > min_keypoints:
                result.track_id = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                result.pred_instances.keypoints[0, :, :] = -10
                result.pred_instances.bboxes *= 0
                result.track_id = -1
        else:
            result.track_id = track_id
        del match_result

    return results, next_id

def kpts_dets(self, img, bboxes, fps):
    if mmpose.__version__[0] == '0':
        bboxes = [{'bbox': bbox} for bbox in bboxes]
        # keypoints detection, outputs list of dicts(bboxes= , keypoints=np.array(17*3, float32))
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results=bboxes,
            format='xyxy',
            return_heatmap=False,
            outputs=None)
        
        # person identification, add keys track_id, area and one_euro
        pose_results, self.next_id = get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=self.mmpose_cfg.TRACKING_THR,
            use_one_euro=True,
            fps=fps)
        for pose_result in pose_results:
            _id = pose_result['track_id']
            xyxy = pose_result['bbox'] #shape: 4,
            bbox = self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(pose_result['keypoints'][None]) # shape: 1,17,3
    else:
        pose_results = inference_topdown(
            self.pose_model,
            img,
            bboxes=bboxes,
            bbox_format='xyxy')
        pose_results, self.next_id = new_get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            tracking_thr=self.mmpose_cfg.TRACKING_THR)
        for pose_result in pose_results:
            _id = pose_result.track_id
            xyxy = pose_result.pred_instances.bboxes[0]
            bbox = self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            kpts = np.zeros((1,17,3), dtype=float)
            kpts[:,:,:2] = pose_result.pred_instances.keypoints
            kpts[:,:,2] = pose_result.pred_instances.keypoint_scores
            self.tracking_results['keypoints'].append(kpts)
    return pose_results


class DetectionModel(object):
    def __init__(self, device, mmpose_cfg):
        
        # ViTPose
        self.mmpose_cfg = mmpose_cfg
        pose_model_cfg = mmpose_cfg.POSE_CONFIG
        pose_model_ckpt = mmpose_cfg.POSE_CHECKPOINT
        print(mmpose.__version__)
        print(f'Using mmpose version {mmpose.__version__}')
        if mmpose.__version__[0] == '0':
            self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        else:
            self.pose_model = init_model('configs/VIT/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py',
                                         'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth',
                                         device=device.lower())
        
        # YOLO
        bbox_model_ckpt = mmpose_cfg.DET_CHECKPOINT
        self.bbox_model = YOLO(bbox_model_ckpt)
        
        self.device = device
        self.initialize_tracking()
        
    def initialize_tracking(self, ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        
    def xyxy_to_cxcys(self, bbox):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
        return np.array([[cx, cy, scale]])
        
    def track(self, img, fps, length):
        
        # bbox detection, output list of np.array(float32)
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=self.mmpose_cfg.BBOX_CONF, save=False, verbose=False)[0].boxes.xyxy.detach().cpu().numpy()
        
        
        """
        bboxes = [{'bbox': bbox} for bbox in bboxes]
        # keypoints detection, outputs list of dicts(bboxes= , keypoints=np.array(17*3, float32))
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results=bboxes,
            format='xyxy',
            return_heatmap=False,
            outputs=None)
        
        # person identification, add keys track_id, area and one_euro
        pose_results, self.next_id = get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=self.mmpose_cfg.TRACKING_THR,
            use_one_euro=True,
            fps=fps)
        """
        pose_results = kpts_dets(self, img, bboxes, fps)
        
        self.frame_id += 1
        self.pose_results_last = pose_results
        
    def process(self, fps):
        for key in ['id', 'frame_id']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        for key in ['keypoints', 'bbox']:
            self.tracking_results[key] = np.concatenate(self.tracking_results[key], axis=0)
            
        output = defaultdict(dict)
        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            output[_id]['features'] = []
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < self.mmpose_cfg.MINIMUM_FRMAES:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox
        
        return output