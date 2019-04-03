#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function)

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

from six import string_types
from base64 import b64encode

import cv2
import os
import math
import sys
import webcolors
import numpy as np

stdout = getattr(sys.stdout, 'buffer', sys.stdout)


def create_blank(height, width, rgb_color):
    blank_img = np.zeros((height, width, 3), np.uint8)
    blank_img[:] = tuple(reversed(rgb_color))
    return blank_img


def padding_blank(image, left, top, right, bottom, color):
    height, width = image.shape[:2]
    pad_img = create_blank(height + top + bottom, width + left + right, color)
    pad_img[top:height + top, left:width + left] = image
    return pad_img


def resize_keep_aspect(image, target_width, target_height, color, interpolation=1):
    height, width = image.shape[:2]
    height_scale = float(target_height) / height
    width_scale = float(target_width) / width
    resize_scale = min(height_scale, width_scale)

    if (width >= height):
        roi_width = target_width
        roi_height = height * resize_scale
        roi_x = 0
        roi_y = int(math.floor((target_height - roi_height) / 2))
    else:
        roi_y = 0
        roi_height = target_height
        roi_width = width * resize_scale
        roi_x = int(math.floor((target_width - roi_width) / 2))

    roi_width = int(math.floor(roi_width))
    roi_height = int(math.floor(roi_height))

    resized_img = cv2.resize(image, (roi_width, roi_height), interpolation=interpolation)
    resized_img = padding_blank(resized_img, roi_x, roi_y, target_width - roi_width - roi_x, target_height - roi_height - roi_y, color)
    return resized_img


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_tile_img(filename_list, args):
    if isinstance(args.space_color, string_types):
        space_color = webcolors.name_to_rgb(args.space_color)
    # interpolation = getattr(cv2, args.interpolation, 1)
    space = args.space
    tile_num = args.tile_num
    interpolation = getattr(cv2, args.interpolation, 1)
    resize_x, resize_y = int(args.size.split('x')[0]), int(args.size.split('x')[1])
    image_list = []
    for filename in filename_list:
        img = cv2.imread(filename)
        if img is None:
            # create blank image
            part_img = np.zeros((resize_y, resize_x, 3), np.uint8)
        else:
            if args.keep_aspect:
                part_img = resize_keep_aspect(img, resize_x, resize_y, space_color, interpolation=interpolation)
            else:
                part_img = cv2.resize(img, (resize_x, resize_y), interpolation=interpolation)
            if space > 0:
                part_img = padding_blank(part_img, space, space, 0, 0, space_color)

        image_list.append(part_img)

    horizontal_image_list = []
    for horizontal in chunks(image_list, tile_num):
        while (len(horizontal) < min(len(image_list), tile_num)):
            height, width = horizontal[0].shape[:2]
            horizontal.append(create_blank(height, width, space_color))
        horizontal_image_list.append(cv2.hconcat(horizontal))

    result_img = cv2.vconcat(horizontal_image_list)
    return result_img


def imgcat_for_iTerm2(imgdata):
    _flag, buf = cv2.imencode('.png', imgdata)
    if os.environ['TERM'].startswith('screen'):
        osc = b'\033Ptmux;\033\033]1337;File='
        st = b'\a\033\\\n'
    else:
        osc = b'\033]1337;File='
        st = b'\a\n'
    stdout.write(b'%ssize=%d;inline=1:%s%s' %
                (osc, len(buf), b64encode(buf), st))
    sys.stdout.flush()

