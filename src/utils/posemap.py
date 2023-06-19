import torch
import numpy as np



def kpoint_to_heatmap(kpoint, shape, sigma):
    """Converts a 2D keypoint to a gaussian heatmap

    Parameters
    ----------
    kpoint: np.array
        2D coordinates of keypoint [x, y].
    shape: tuple
        Heatmap dimension (HxW).
    sigma: float
        Variance value of the gaussian.

    Returns
    -------
    heatmap: np.array
        A gaussian heatmap HxW.
    """
    map_h = shape[0]
    map_w = shape[1]
    if np.any(kpoint > 0):
        x, y = kpoint
        # x = x * map_w / 384.0
        # y = y * map_h / 512.0
        xy_grid = np.mgrid[:map_w, :map_h].transpose(2, 1, 0)
        heatmap = np.exp(-np.sum((xy_grid - (x, y)) ** 2, axis=-1) / sigma ** 2)
        heatmap /= (heatmap.max() + np.finfo('float32').eps)
    else:
        heatmap = np.zeros((map_h, map_w))
    return torch.Tensor(heatmap)

def get_coco_body25_mapping():
    #left numbers are coco format while right numbers are body25 format
    return {
        0:0,
        1:1,
        2:2,
        3:3,
        4:4,
        5:5,
        6:6,
        7:7,
        8:9,
        9:10,
        10:11,
        11:12,
        12:13,
        13:14,
        14:15,
        15:16,
        16:17,
        17:18
    }