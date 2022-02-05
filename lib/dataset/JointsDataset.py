# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, coord_representation='heatmap', simdr_split_ratio=1):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

        # simdr related
        self.coord_representation = coord_representation
        self.simdr_split_ratio = simdr_split_ratio
        self.loss = cfg.LOSS.TYPE
        assert self.coord_representation in ['simdr', 'sa-simdr', 'heatmap'], 'only simdr or sa-simdr or heamtap supported'

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        # print(image_file)
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            # print(data_numpy)
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)
        
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }
        if self.coord_representation == 'simdr':
            joints_split = joints.copy()
            joints_split = np.around(joints_split * self.simdr_split_ratio)
            joints_split = joints_split.astype(np.int64) 
            target_weight,filtered_joints = self.filter_target_simdr(joints_split.copy(), joints_vis, self.image_size*self.simdr_split_ratio)
            return input, filtered_joints[:,0:2], target_weight, meta
        elif self.coord_representation == 'sa-simdr':
            target_x, target_y, target_weight = self.generate_sa_simdr(joints, joints_vis)
            target_x = torch.from_numpy(target_x)
            target_y = torch.from_numpy(target_y)
            target_weight = torch.from_numpy(target_weight)
            return input, target_x, target_y, target_weight, meta            
        elif self.coord_representation == 'heatmap':
            return input, target, target_weight, meta

    def generate_sa_simdr(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target_x = np.zeros((self.num_joints,
                            int(self.image_size[0]*self.simdr_split_ratio)),
                            dtype=np.float32)
        target_y = np.zeros((self.num_joints,
                            int(self.image_size[1]*self.simdr_split_ratio)),
                            dtype=np.float32)                              

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            target_weight[joint_id] = \
                self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
            if target_weight[joint_id] == 0:
                continue

            mu_x = joints[joint_id][0] * self.simdr_split_ratio
            mu_y = joints[joint_id][1] * self.simdr_split_ratio
            
            x = np.arange(0, int(self.image_size[0]*self.simdr_split_ratio), 1, np.float32)
            y = np.arange(0, int(self.image_size[1]*self.simdr_split_ratio), 1, np.float32)

            v = target_weight[joint_id]
            if v > 0.5:
                target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * self.sigma ** 2)))/(self.sigma*np.sqrt(np.pi*2))
                target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * self.sigma ** 2)))/(self.sigma*np.sqrt(np.pi*2))
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target_x, target_y, target_weight  

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected   

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32) # [size]
            y = x[:, np.newaxis] # [size,1]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)) # [size] 

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def filter_target_simdr(self, joints, joints_vis, image_size):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :param image_size: image_size
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0].copy()

        # detect abnormal coords and make the weight 0
        for joint_id in range(self.num_joints):
            if joints[joint_id][1] < 0:
                target_weight[joint_id] = 0
                joints[joint_id][1]=0
            elif joints[joint_id][1] >= image_size[1]:
                target_weight[joint_id] = 0
                joints[joint_id][1] = image_size[1] - 1
            if joints[joint_id][0] < 0:
                target_weight[joint_id] = 0
                joints[joint_id][0] = 0
            elif joints[joint_id][0] >= image_size[0]:
                target_weight[joint_id] = 0
                joints[joint_id][0] = image_size[0] - 1

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target_weight,joints  

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= (self.image_size[0]) or ul[1] >= self.image_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight
        

