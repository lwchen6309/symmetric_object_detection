from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from glob import glob
import os
import torch
from flip_rotation import flip_mask_with_symmetric_center
from time import time


def compute_iou(mask1, mask2):
    """
    Compute the intersection over union (IoU) score between two masks.

    Parameters:
    mask1 (ndarray): A boolean mask.
    mask2 (ndarray): A boolean mask.

    Returns:
    The IoU score between the two masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def symmetric_images_detection(model_path, image_paths, 
                               iou_threshold=0.8, area_ratio_threshold=0.1, thetas=np.arange(0, 180, 30)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=model_path).to(device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    filtered_segs = []
    filtered_axes = []
    max_ious = []
    max_thetas = []
    for image_path in image_paths:
        print('Process image: %s'%image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        masks = mask_generator.generate(image)

        masks_mirror_ious = []
        masks_mirror_thetas = []
        symmetric_axes = []
        for d in masks:
            seg_masks = d['segmentation'].astype(np.float32)
            bbox = d['bbox']
            center = [bbox[1] + bbox[3]//2, bbox[0] + bbox[2]//2]
            ious = []
            axes = []
            for i, theta in enumerate(thetas):
                mirror_mask, symmetric_axis = flip_mask_with_symmetric_center(seg_masks, center, theta)
                iou = compute_iou(seg_masks > 0.5, mirror_mask > 0.5)
                ious.append(iou)
                axes.append(symmetric_axis)
            ious = np.array(ious)
            argmax_theta = np.argmax(ious)
            masks_mirror_ious.append(ious[argmax_theta])
            masks_mirror_thetas.append(argmax_theta)
            symmetric_axes.append(axes[argmax_theta])

        masks_mirror_ious = np.array(masks_mirror_ious)
        masks_mirror_thetas = np.array(masks_mirror_thetas)
        filtered_seg = []
        filtered_ax = []
        inv_total_area = 1 / np.prod(image.shape[:2])
        for _mask, iou, theta, axis in zip(masks, masks_mirror_ious, masks_mirror_thetas, symmetric_axes):
            area_ratio = _mask['area'] * inv_total_area
            if iou < iou_threshold or area_ratio < area_ratio_threshold:
                continue
            filtered_seg.append(_mask['segmentation'])
            filtered_ax.append(axis)
        filtered_segs.append(filtered_seg)
        filtered_axes.append(filtered_ax)
        max_ious.append(masks_mirror_ious)
        max_thetas.append(masks_mirror_thetas)
    return filtered_segs, filtered_axes, max_ious, max_thetas


if __name__ == '__main__':
    model_path = "./models/sam_vit_h_4b8939.pth"
    image_paths = glob('./images/*')
    filtered_segs, filtered_axes, max_ious, max_thetas = symmetric_images_detection(model_path, image_paths, 
        thetas=np.arange(0, 180, 30), 
        iou_threshold=0.8, 
        area_ratio_threshold=0.1)
    
    for image_path, filtered_seg, filtered_axis, max_theta in zip(image_paths, filtered_segs, filtered_axes, max_thetas):
        if len(filtered_seg) == 0:
            continue
        file_basename = os.path.basename(image_path).split('.')[0]
        fig, axes = plt.subplots(1, len(filtered_seg) + 1)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(image)
        axes[0].axis('off')
        for seg, sym_axs, axis in zip(filtered_seg, filtered_axis, axes[1:]):
            axis.imshow(seg, alpha=0.5)
            axis.imshow(sym_axs, alpha=0.5)
            axis.axis('off')
            axis.set_frame_on(False)
        
        plt.savefig(os.path.join("./outputs", f"{file_basename}_output.png"), bbox_inches='tight', pad_inches=0)
        plt.close()