import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def shift_image(image, shift_distance):
    # Create output array with the same shape as input
    shifted = np.empty_like(image)
    
    # Fill shifted array with shifted image values
    if shift_distance[0] >= 0:
        shifted[shift_distance[0]:, :] = image[:-shift_distance[0], :]
        shifted[:shift_distance[0], :] = 0
    else:
        shifted[:shift_distance[0], :] = image[-shift_distance[0]:, :]
        shifted[shift_distance[0]:, :] = 0
        
    if shift_distance[1] >= 0:
        shifted[:, shift_distance[1]:] = shifted[:, :-shift_distance[1]]
        shifted[:, :shift_distance[1]] = 0
    else:
        shifted[:, :shift_distance[1]] = shifted[:, -shift_distance[1]:]
        shifted[:, shift_distance[1]:] = 0
        
    return shifted


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


# Create an ellipse
ylen = 1000
xlen = 2000
x, y = np.meshgrid(np.linspace(-100, 100, xlen), np.linspace(-100, 100, ylen))
offset = (100, 500)
center = (ylen//2 + offset[0], xlen//2 + offset[1])
theta = 125
ellipse = ((x / 30) ** 2 + (y / 20) ** 2 <= 1)
ellipse = ellipse.astype(np.float32)
ellipse = shift_image(ellipse, [offset[0], offset[1]])

final_ellipse = flip_mask_with_symmetric_center(ellipse, center, theta)

# Plot each step of the process
fig, axs = plt.subplots(2)
axs[0].imshow(ellipse, alpha=0.5)
axs[0].set_title('Original ellipse')
axs[0].imshow(final_ellipse, alpha=0.5)
axs[0].set_title('Final ellipse')
plt.show()
