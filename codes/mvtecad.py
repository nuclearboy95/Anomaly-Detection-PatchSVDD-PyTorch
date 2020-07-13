import numpy as np
from PIL import Image
from imageio import imread
from glob import glob
from sklearn.metrics import roc_auc_score
import os

DATASET_PATH = '/path/to/the/dataset'


__all__ = ['objs', 'set_root_path',
           'get_x', 'get_x_standardized',
           'detection_auroc', 'segmentation_auroc']

objs = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
        'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
        'transistor', 'wood', 'zipper']


def resize(image, shape=(256, 256)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))


def bilinears(images, shape) -> np.ndarray:
    import cv2
    N = images.shape[0]
    new_shape = (N,) + shape
    ret = np.zeros(new_shape, dtype=images.dtype)
    for i in range(N):
        ret[i] = cv2.resize(images[i], dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
    return ret


def gray2rgb(images):
    tile_shape = tuple(np.ones(len(images.shape), dtype=int))
    tile_shape += (3,)

    images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
    # print(images.shape)
    return images


def set_root_path(new_path):
    global DATASET_PATH
    DATASET_PATH = new_path


def get_x(obj, mode='train'):
    fpattern = os.path.join(DATASET_PATH, f'{obj}/{mode}/*/*.png')
    fpaths = sorted(glob(fpattern))

    if mode == 'test':
        fpaths1 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
        fpaths2 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))

        images1 = np.asarray(list(map(imread, fpaths1)))
        images2 = np.asarray(list(map(imread, fpaths2)))
        images = np.concatenate([images1, images2])

    else:
        images = np.asarray(list(map(imread, fpaths)))

    if images.shape[-1] != 3:
        images = gray2rgb(images)
    images = list(map(resize, images))
    images = np.asarray(images)
    return images


def get_x_standardized(obj, mode='train'):
    x = get_x(obj, mode=mode)
    mean = get_mean(obj)
    return (x.astype(np.float32) - mean) / 255


def get_label(obj):
    fpattern = os.path.join(DATASET_PATH, f'{obj}/test/*/*.png')
    fpaths = sorted(glob(fpattern))
    fpaths1 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
    fpaths2 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))

    Nanomaly = len(fpaths1)
    Nnormal = len(fpaths2)
    labels = np.zeros(Nanomaly + Nnormal, dtype=np.int32)
    labels[:Nanomaly] = 1
    return labels


def get_mask(obj):
    fpattern = os.path.join(DATASET_PATH, f'{obj}/ground_truth/*/*.png')
    fpaths = sorted(glob(fpattern))
    masks = np.asarray(list(map(lambda fpath: resize(imread(fpath), (256, 256)), fpaths)))
    Nanomaly = masks.shape[0]
    Nnormal = len(glob(os.path.join(DATASET_PATH, f'{obj}/test/good/*.png')))

    masks[masks <= 128] = 0
    masks[masks > 128] = 255
    results = np.zeros((Nanomaly + Nnormal,) + masks.shape[1:], dtype=masks.dtype)
    results[:Nanomaly] = masks

    return results


def get_mean(obj):
    images = get_x(obj, mode='train')
    mean = images.astype(np.float32).mean(axis=0)
    return mean


def detection_auroc(obj, anomaly_scores):
    label = get_label(obj)  # 1: anomaly 0: normal
    auroc = roc_auc_score(label, anomaly_scores)
    return auroc


def segmentation_auroc(obj, anomaly_maps):
    gt = get_mask(obj)
    gt = gt.astype(np.int32)
    gt[gt == 255] = 1  # 1: anomaly

    anomaly_maps = bilinears(anomaly_maps, (256, 256))
    auroc = roc_auc_score(gt.flatten(), anomaly_maps.flatten())
    return auroc

