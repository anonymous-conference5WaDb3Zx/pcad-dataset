#!/usr/bin/env python3
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import yaml

import bpy
import numpy as np
from PIL import Image

import math
import re
from typing import Optional, Dict, Any, Union
import os
import cv2
import inspect

def blender_center_crop(scene):
    # compute (shifted) center
    center = (scene.render.resolution_x / 2, scene.render.resolution_y / 2)

    # NOTE: shifting is inverted for x-direction because x-axis goes to the left from the perspective
    # the viewport.
    center_shifted = (math.floor(center[0] - scene.camera.data.shift_x * scene.render.resolution_x),
                      math.floor(center[1] + scene.camera.data.shift_y * scene.render.resolution_y))

    # get largest square
    size_half = min([center_shifted[1], scene.render.resolution_y - center_shifted[1]])
    size = 2 * size_half
    pos_left_upper = (math.floor(center_shifted[0] - size / 2),
                      math.floor(center_shifted[1] - size / 2))
    pos_right_lower = (math.floor(center_shifted[0] + size / 2),
                       math.floor(center_shifted[1] + size / 2))

    # make sure that square has expected dimensions
    assert pos_right_lower[0] - pos_left_upper[0] == pos_right_lower[1] - pos_left_upper[1]
    assert pos_right_lower[1] == scene.render.resolution_y or pos_left_upper[1] == 0

    return pos_left_upper, pos_right_lower

def frame_size(image_set):
    print(image_set)
    frame_fn = f"./data/images/2700642-training/no-defect/frame-snapshot_product-name=mguard_product-defect=none_camera-rotation=050_table-rotation=000_frame=0.png"
    assert os.path.exists(frame_fn)
    img = cv2.imread(frame_fn)
    return img.shape[0], img.shape[1]


def convert_deg_to_rad(vec):
    return R.from_euler('xyz', vec, degrees=True).as_euler('xyz', degrees=False)

def load_cam_settings(camera_name, fn_settings):
    with open(fn_settings, "r") as fh:
        cam_settings = list(yaml.safe_load_all(fh))[0]
        # assert isinstance(cam_settings, dict)
        for k, v in cam_settings.items():
            print(k, v)
            bpy.data.cameras[camera_name][k] = v

def cv_to_blender(vec):
    return np.array([[[vec[..., 0], -vec[..., 1], -vec[..., 2]]]])

def image_overlay(img: Image.Image, overlay: Image.Image, alpha: float=0.5):
    overlay = overlay.convert('RGBA')
    overlay.putalpha(127)
    # overlay = reduce_alpha(overlay, alpha)
    img.paste(overlay, (0, 0), overlay)
    return img

def reduce_alpha(img: Image.Image, alpha_reduction=0.5):
    pixdata = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixdata[x, y] = tuple([int(math.floor(el * alpha_reduction)) for el in pixdata[x, y]])
    return img

def filename_property(fn, keyword, delimiter="=", use_terminator=None) -> Optional[int]:
    if use_terminator:
        p = re.compile(f'.*{keyword}{delimiter}([^_]*)_[^_]*')
        m = p.match(fn)
        if not m:
            return None
        return m.group(1)

    else:
        p = re.compile(f'.*{keyword}{delimiter}([0-9]*)')
        m = p.match(fn)
        if not m:
            return None
        return int(m.group(1))


def construct_property_filename(keywords: Dict[str, int], prefix: str=None, delimiter="=", file_extension: Optional[str]=None):
    fn = f"{prefix}_"
    fn += "_".join([f"{k}{delimiter}{v}" for k, v in keywords.items()])
    if file_extension:
        assert file_extension[0] == '.'
        fn += file_extension
    return fn

def has_prefix(fn: Union[str, Path], prefix: str):
    if isinstance(fn, Path):
        fn = str(fn)
    return fn.startswith(prefix)

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def report_unexpected(var: Any):
    return f"{retrieve_name(var)} has unexpected valued: {var}"

def is_set(var: Any|None):
    return var is not None

def create_filename_pack(fn_lists, **kwargs):
    key_lists = []
    fn_dict = {}

    for fn_list in fn_lists:
        key_list = [filename_property(fn, keyword=kwargs['keyword'], delimiter=kwargs['delimiter']) for fn in fn_list]
        print(key_list)
        assert None not in key_list
        key_lists.append(key_list)
        for fn, key in zip(fn_list, key_list):
            if key not in fn_dict:
                fn_dict[key] = []
            fn_dict[key].append(fn)

    import pprint
    l = []
    for k, v in sorted(fn_dict.items()):
        if len(v) != len(fn_lists):
            print(f"File missing in\n{pprint.pformat(v)}. Dropping!")
            continue
        l.append(v)

    return l
