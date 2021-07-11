# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, NMTCritierion, NMTNORMCritierion, KLDiscretLoss
from core.function import validate_heatmap, validate_simdr, validate_sa_simdr
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )               
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    if cfg.LOSS.TYPE == 'JointsMSELoss':
        criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    elif cfg.LOSS.TYPE == 'NMTCritierion':
        criterion = NMTCritierion(label_smoothing=cfg.LOSS.LABEL_SMOOTHING).cuda()
    elif cfg.LOSS.TYPE == 'NMTNORMCritierion':
        criterion = NMTNORMCritierion(label_smoothing=cfg.LOSS.LABEL_SMOOTHING).cuda()
    elif cfg.LOSS.TYPE == 'KLDiscretLoss':
        criterion = KLDiscretLoss().cuda()           
    else:
        criterion = L1JointLocationLoss().cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        cfg.MODEL.COORD_REPRESENTATION,
        cfg.MODEL.SIMDR_SPLIT_RATIO
    )    

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    if cfg.MODEL.COORD_REPRESENTATION == 'simdr':
        validate_simdr(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir)    
    elif cfg.MODEL.COORD_REPRESENTATION == 'sa-simdr':
        validate_sa_simdr(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir)             
    elif cfg.MODEL.COORD_REPRESENTATION == 'heatmap':
        validate_heatmap(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
