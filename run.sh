#!/bin/bash

# use multi-GPU to pretrain  GPU: [0, 2]
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6667 train_pair.py --config_file configs/pretrain_transoss.yml MODEL.DIST_TRAIN True

# use multi-GPU to train  GPU: [0, 2]
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6667 train.py --config_file configs/cmfgo_transoss.yml MODEL.DIST_TRAIN True

# use single GPU to train
python train.py --config_file configs/cmfgo_transoss.yml

# validation
python test.py --config_file configs/cmfgo_transoss.yml MODEL.DEVICE_ID "('2')"  TEST.WEIGHT 'weights/CMFGO.pth'
python test.py --config_file configs/cmfgo_transoss.yml MODEL.DEVICE_ID "('2')"  TEST.WEIGHT 'weights/CMFGO_unpair.pth'

# validation S2O
python test.py --config_file configs/cmfgo_transoss_S2O.yml MODEL.DEVICE_ID "('2')"  TEST.WEIGHT 'weights/CMFGO.pth'

# validation O2S
python test.py --config_file configs/cmfgo_transoss_O2S.yml MODEL.DEVICE_ID "('2')"  TEST.WEIGHT 'weights/CMFGO.pth'

