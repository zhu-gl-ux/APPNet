# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image
from .base import BaseDataset


class Cocostuff(BaseDataset):
    """Cityscapes dataset generator"""

    def __init__(self, opt):
        super().__init__(opt)

        # There are some ignored classes in this dataset
        ignore_label = 255
        self.ignore_label = ignore_label
        self.mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                        78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96,
                        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                        113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
                        129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
                        145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                        161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
                        177, 178, 179, 180, 181, 182]

        self.label2color = {0:(0, 192, 64),
                        1:(0, 192, 64),
                        2:(0, 64, 96),
                        3:(128, 192, 192),
                        4:(0, 64, 64),
                        5:(0, 192, 224),
                        6:(0, 192, 192),
                        7:(128, 192, 64),
                        8:(0, 192, 96),
                        9:(128, 192, 64),
                        10:(128, 32, 192),
                        11:(0, 0, 224),
                        12:(0, 0, 64),
                        13:(0, 160, 192),
                        14:(128, 0, 96),
                        15:(128, 0, 192),
                        16:(0, 32, 192),
                        17:(128, 128, 224),
                        18:(0, 0, 192),
                        19:(128, 160, 192),
                        20:(128, 128, 0),
                        21:(128, 0, 32),
                        22:(128, 32, 0),
                        23:(128, 0, 128),
                        24:(64, 128, 32),
                        25:(0, 160, 0),
                        26:(0, 0, 0),
                        27:(192, 128, 160),
                        28:(0, 32, 0),
                        29:(0, 128, 128),
                        30:(64, 128, 160),
                        31:(128, 160, 0),
                        32:(0, 128, 0),
                        33:(192, 128, 32),
                        34:(128, 96, 128),
                        35:(0, 0, 128),
                        36:(64, 0, 32),
                        37:(0, 224, 128),
                        38:(128, 0, 0),
                        39:(192, 0, 160),
                        40:(0, 96, 128),
                        41:(128, 128, 128),
                        42:(64, 0, 160),
                        43:(128, 224, 128),
                        44:(128, 128, 64),
                        45:(192, 0, 32),
                        46:(128, 96, 0),
                        47:(128, 0, 192),
                        48:(0, 128, 32),
                        49:(64, 224, 0),
                        50:(0, 0, 64),
                        51:(128, 128, 160),
                        52:(64, 96, 0),
                        53:(0, 128, 192),
                        54:(0, 128, 160),
                        55:(192, 224, 0),
                        56:(0, 128, 64),
                        57:(128, 128, 32),
                        58:(192, 32, 128),
                        59:(0, 64, 192),
                        60:(0, 0, 32),
                        61:(64, 160, 128),
                        62:(128, 64, 64),
                        63:(128, 0, 160),
                        64:(64, 32, 128),
                        65:(128, 192, 192),
                        66:(0, 0, 160),
                        67:(192, 160, 128),
                        68:(128, 192, 0),
                        69:(128, 0, 96),
                        70:(192, 32, 0),
                        71:(128, 64, 128),
                        72:(64, 128, 96),
                        73:(64, 160, 0),
                        74:(0, 64, 0),
                        75:(192, 128, 224),
                        76:(64, 32, 0),
                        77:(0, 192, 128),
                        78:(64, 128, 224),
                        79:(192, 160, 0),
                        80:(0, 192, 0),
                        81:(192, 128, 96),
                        82:(192, 96, 128),
                        83:(0, 64, 128),
                        84:(64, 0, 96),
                        85:(64, 224, 128),
                        86:(128, 64, 0),
                        87:(192, 0, 224),
                        88:(64, 96, 128),
                        89:(128, 192, 128),
                        90:(64, 0, 224),
                        91:(192, 224, 128),
                        92:(128, 192, 64),
                        93:(192, 0, 96),
                        94:(192, 96, 0),
                        95:(128, 64, 192),
                        96:(0, 128, 96),
                        97:(0, 224, 0),
                        98:(64, 64, 64),
                        99:(128, 128, 224),
                        100:(0, 96, 0),
                        101:(64, 192, 192),
                        102:(0, 128, 224),
                        103:(128, 224, 0),
                        104:(64, 192, 64),
                        105:(128, 128, 96),
                        106:(128, 32, 128),
                        107:(64, 0, 192),
                        108:(0, 64, 96),
                        109:(0, 160, 128),
                        110:(192, 0, 64),
                        111:(128, 64, 224),
                        112:(0, 32, 128),
                        113:(192, 128, 192),
                        114:(0, 64, 224),
                        115:(128, 160, 128),
                        116:(192, 128, 0),
                        117:(128, 64, 32),
                        118:(128, 32, 64),
                        119:(192, 0, 128),
                        120:(64, 192, 32),
                        121:(0, 160, 64),
                        122:(64, 0, 0),
                        123:(192, 192, 160),
                        124:(0, 32, 64),
                        125:(64, 128, 128),
                        126:(64, 192, 160),
                        127:(128, 160, 64),
                        128:(64, 128, 0),
                        129:(192, 192, 32),
                        130:(128, 96, 192),
                        131:(64, 0, 128),
                        132:(64, 64, 32),
                        133:(0, 224, 192),
                        134:(192, 0, 0),
                        135:(192, 64, 160),
                        136:(0, 96, 192),
                        137:(192, 128, 128),
                        138:(64, 64, 160),
                        139:(128, 224, 192),
                        140:(192, 128, 64),
                        141:(192, 64, 32),
                        142:(128, 96, 64),
                        143:(192, 0, 192),
                        144:(0, 192, 32),
                        145:(64, 224, 64),
                        146:(64, 0, 64),
                        147:(128, 192, 160),
                        148:(64, 96, 64),
                        149:(64, 128, 192),
                        150:(0, 192, 160),
                        151:(192, 224, 64),
                        152:(64, 128, 64),
                        153:(128, 192, 32),
                        154:(192, 32, 192),
                        155:(64, 64, 192),
                        156:(0, 64, 32),
                        157:(64, 160, 192),
                        158:(192, 64, 64),
                        159:(128, 64, 160),
                        160:(64, 32, 192),
                        161:(192, 192, 192),
                        162:(0, 64, 160),
                        163:(192, 160, 192),
                        164:(192, 192, 0),
                        165:(128, 64, 96),
                        166:(192, 32, 64),
                        167:(192, 64, 128),
                        168:(64, 192, 96),
                        169:(64, 160, 64),
                        170:(64, 64, 0),
                            }
        self.label_reading_mode = cv2.IMREAD_GRAYSCALE

    def image2class(self, labelmap):
        # ret = np.ones_like(labelmap) * 255
        # for idx, label in enumerate(self.mapping):
        #     ret[labelmap == label] = idx
        # labelmap = np.array(ret)
        # encoded_labelmap = labelmap - 1

        return labelmap

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    # def __getitem__(self, index):
    #     item = self.files[index]
    #     name = item["name"]
    #     image_path = os.path.join(self.root, item['img'])
    #     label_path = os.path.join(self.root, item['label'])
    #     image = cv2.imread(
    #         image_path,
    #         cv2.IMREAD_COLOR
    #     )
    #     label = np.array(
    #         Image.open(label_path).convert('P')
    #     )
    #     label = self.encode_label(label)
    #     label = self.reduce_zero_label(label)
    #     size = label.shape
