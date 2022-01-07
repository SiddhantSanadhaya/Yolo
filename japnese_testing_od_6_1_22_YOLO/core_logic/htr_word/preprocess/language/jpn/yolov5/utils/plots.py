# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""

import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont
from core_logic.htr_word.preprocess.language.jpn.yolov5.utils.general import (clip_coords, xywh2xyxy, xyxy2xywh)



def save_one_box(xyxy, im, imc_copy, gain=1.0, pad=0, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop

    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    # image = np_data[start_y_text:end_y_text,start_x_text:end_x_text]
    cv2.rectangle(imc_copy,(int(xyxy[0, 0]),xyxy[0, 1])),(int(xyxy[0, 2]),int(xyxy[0, 3])),(255,255,0),2)
    # if save:
    #     file.parent.mkdir(parents=True, exist_ok=True)  # make directory
    #     cv2.imwrite(str(increment_path(file).with_suffix('.jpg')), crop)
    return crop,start_x_text, imc_copy
