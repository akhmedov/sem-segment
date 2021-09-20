import os
import numpy as np

LABEL_LEGEND = {

    0: {
        'name': 'unlabeled',
        'category': 'void',
        'color': (0, 0, 0)
    },

    1: {
        'name': 'ego-vehicle',
        'category': 'void',
        'color': (0, 0, 0)
    },

    2: {
        'name': 'rectification-border',
        'category': 'void',
        'color': (0, 0, 0)
    },

    3: {
        'name': 'out-of-roi',
        'category': 'void',
        'color': (0, 0, 0)
    },

    4: {
        'name': 'static',
        'category': 'void',
        'color': (0, 0, 0)
    },

    5: {
        'name': 'dynamic',
        'category': 'void',
        'color': (111, 74,  0)
    },

    6: {
        'name': 'ground',  # refuge-island
        'category': 'void',
        'color': (81, 0, 81)
    },

    7: {
        'name': 'road',
        'category': 'flat',
        'color': (128, 64, 128)
    },

    8: {
        'name': 'sidewalk',
        'category': 'flat',
        'color': (244, 35, 232)
    },

    9: {
        'name': 'parking',
        'category': 'flat',
        'color': (250, 170, 160)
    },

    10: {
        'name': 'rail-track',
        'category': 'flat',
        'color': (230, 150, 140)
    },

    11: {
        'name': 'building',
        'category': 'construction',
        'color': (70, 70, 70)
    },

    12: {
        'name': 'wall',
        'category': 'construction',
        'color': (102, 102, 156)
    },

    13: {
        'name': 'fence',
        'category': 'construction',
        'color': (190, 153, 153)
    },

    14: {
        'name': 'guard-rail',
        'category': 'construction',
        'color': (180, 165, 180)
    },

    15: {
        'name': 'bridge',
        'category': 'construction',
        'color': (150, 100, 100)
    },

    16: {
        'name': 'tunnel',
        'category': 'construction',
        'color': (150, 120, 90)
    },

    17: {
        'name': 'pole',
        'category': 'object',
        'color': (153, 153, 153)
    },

    18: {
        'name': 'polegroup',
        'category': 'object',
        'color': (153, 153, 153)
    },

    19: {
        'name': 'traffic-light',
        'category': 'object',
        'color': (250, 170, 30)
    },

    20: {
        'name': 'traffic-sign',
        'category': 'object',
        'color': (220, 220, 0)
    },

    21: {
        'name': 'vegetation',
        'category': 'nature',
        'color': (107, 142, 35)
    },

    22: {
        'name': 'terrain',
        'category': 'nature',
        'color': (152, 251, 152)
    },

    23: {
        'name': 'sky',
        'category': 'sky',
        'color': (70, 130, 180)
    },

    24: {
        'name': 'person',
        'category': 'human',
        'color': (220, 20, 60)
    },

    25: {
        'name': 'rider',
        'category': 'human',
        'color': (255, 0, 0)
    },

    26: {
        'name': 'car',
        'category': 'vehicle',
        'color': (0, 0, 142)
    },

    27: {
        'name': 'truck',
        'category': 'vehicle',
        'color': (0, 0, 70)
    },

    28: {
        'name': 'bus',
        'category': 'vehicle',
        'color': (0, 60, 100)
    },

    29: {
        'name': 'caravan',
        'category': 'vehicle',
        'color': (0, 0, 90)
    },

    30: {
        'name': 'trailer',
        'category': 'vehicle',
        'color': (0, 0, 110)
    },

    31: {
        'name': 'train',
        'category': 'vehicle',
        'color': (0, 80, 100)
    },

    32: {
        'name': 'motorcycle',
        'category': 'vehicle',
        'color': (0, 0, 230)
    },

    33: {
        'name': 'bicycle',
        'category': 'vehicle',
        'color': (119, 11, 32)
    },

    -1: {
        'name': 'license-plate',
        'category': 'vehicle',
        'color': (0, 0, 142)
    }
}

CLASS_NUMBER = len(LABEL_LEGEND)
COLOR_SET_RGB = dict([(value['color'], value['name']) for _, value in LABEL_LEGEND.items()])


def list_classes(image: np.array):
    pixels = image.reshape(-1, image.shape[2])
    colors = np.unique(pixels, axis=0).tolist()
    classes = list()
    for item in colors:
        item = tuple(item)
        item = COLOR_SET_RGB.get(item, item)
        classes.append(item)
    return classes


def match_filenames(image_path_list: list, target_path_list: list, pattern=None):
    updated_target = list()
    for path in image_path_list:
        image_index = os.path.basename(path).replace('_leftImg8bit.png', '', 1)
        image_index = os.path.basename(image_index).replace('_rightImg8bit.png', '', 1)
        related = [gt_name for gt_name in target_path_list if image_index in gt_name]
        if pattern:
            related_and_matched = [item_path for item_path in related if pattern in item_path]
        else:
            related_and_matched = related
        updated_target.append(related_and_matched[0])
    return image_path_list, updated_target
