from functools import partial
import argparse
import cv2
import glob
import json
import logging
import mmcv
import os
import re
import random
import sys
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
from typing import List, Optional
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.nn import init
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms
from torchvision.ops import box_convert

sys.path.append("../dinov2")
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import extract_features

from dataloaders import InsDetDataset_v4
from finetune_DINOv2 import TrainerWithTriplet
from modules import eval
from visualization import batched_nms, imshow_gt_det_bboxes


# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
logger = logging.getLogger("GroundingDINO_DINOv2")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Deterministic setting for reproduciablity.
random.seed(77)
torch.manual_seed(77)
torch.cuda.manual_seed_all(77)
cudnn.deterministic = True

# Enable cuDNN benchmark mode to select the fastest convolution algorithm.
cudnn.enable = True
cudnn.benchmark = True
torch.cuda.set_device(0)
torch.set_num_threads(1)

print("success until here")