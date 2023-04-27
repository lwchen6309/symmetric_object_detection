from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from glob import glob


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


def flip_mask_with_symmetric_center(mask, center, theta):
    """
    Flip a binary mask with respect to its symmetric center and given rotation angle.
    
    Arguments:
    mask -- 2D numpy array, binary mask to be flipped
    center -- tuple, center of the symmetric axis
    theta -- float, rotation angle (in degrees)
    
    Returns:
    flipped_mask -- 2D numpy array, flipped binary mask
    """
    
    # Compute shift required to center the bounding box in the image
    center = np.array(center).astype(int)
    shift = tuple(center - (np.array(mask.shape) / 2).astype(int))
    neg_shift = tuple([-x for x in shift])
    
    # Shift the image
    shifted_mask = np.roll(mask, neg_shift, axis=(0, 1))
    
    # Flip the image with respect to the symmetric axis
    flipped_mask = rotate(shifted_mask, -theta, reshape=False)
    flipped_mask = np.fliplr(flipped_mask)
    flipped_mask = rotate(flipped_mask, theta, reshape=False)
    
    # Shift the image back to its original position
    flipped_mask = np.roll(flipped_mask, shift, axis=(0, 1))
    
    return flipped_mask


def symmetric_images_detection(model_path, thetas = np.arange(0, 180, 30)):  
    sam = sam_model_registry["vit_h"](checkpoint=model_path).to("cuda")
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    filtered_segs = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        masks = mask_generator.generate(image)

        masks_mirror_ious = []
        for d in masks:
            seg_masks = d['segmentation'].astype(np.float32)
            bbox = d['bbox']
            center = [bbox[1] + bbox[3]//2, bbox[0] + bbox[2]//2]
            ious = []
            for i, theta in enumerate(thetas):
                mirror_mask = flip_mask_with_symmetric_center(seg_masks, center, theta)
                iou = compute_iou(seg_masks > 0.5, mirror_mask > 0.5)
                ious.append(iou)
            ious = np.array(ious)
            argmax_theta = np.argmax(ious)
            masks_mirror_ious.append(ious[argmax_theta])

        masks_mirror_ious = np.array(masks_mirror_ious)
        filtered_seg = []
        inv_total_area = 1 / np.prod(image.shape[:2])
        for _mask, iou in zip(masks, masks_mirror_ious):
            area_ratio = _mask['area'] * inv_total_area
            if iou < 0.8 or area_ratio < 0.1:
                continue
            filtered_seg.append(_mask['segmentation'])
        filtered_segs.append(filtered_seg)
    return filtered_segs


if __name__ == '__main__':
    model_path = "./models/sam_vit_h_4b8939.pth"
    image_paths = glob('./images/*')
    filtered_segs = symmetric_images_detection(model_path, thetas = np.arange(0, 180, 30))

    for image_path, filtered_seg in zip(image_paths, filtered_segs):
        if len(filtered_seg) == 0:
            continue
        fig, axes = plt.subplots(1, len(filtered_seg) + 1)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[0].imshow(image)
        for seg, axis in zip(filtered_seg, axes[1:]):
            axis.imshow(seg)
    plt.show()