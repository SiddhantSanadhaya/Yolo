import torch
import os
import torch.backends.cudnn as cudnn
# from  cfg.hmr_logger import Hmr_logger
# from core_logic.model_manager import maths_detect_fn as model
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import (non_max_suppression, scale_coords)
from yolov5.utils.plots import save_one_box
from matplotlib import pyplot as plt
import cv2
from yolov5.models.common import DetectMultiBackend
@torch.no_grad()

def model():
    weights = 'D:/Rawattech/yolov5/best_300.pt'
    model_loaded = DetectMultiBackend(weights)
    return model_loaded
# model = model()

def preprocess(np_data):
    try:
        model.model.float()
        dataset = LoadImages(np_data)
        im, im0s = dataset.__next__()
        im = torch.from_numpy(im)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)
        # NMS
        pred = non_max_suppression(pred)[0]
        imc = im0s.copy()# for save_crop
        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0s.shape).round()

            for *xyxy, conf, cls in reversed(pred):
                cropped_img = save_one_box(xyxy, imc, BGR=True)
                plt.imshow(cropped_img)
                plt.show()

        return cropped_img
        
    except Exception as e:
        # Hmr_logger.log(Hmr_logger.error,"preprocessing_util : preprocess : failure : {}".format(e))
        # timer.endOp()
        raise e


# img = cv2.imread(r"D:\Rawattech\yolov5\2020D02431S143S1_01_HAQ143S000691r.jpg")
# cropped_img = preprocess(img)