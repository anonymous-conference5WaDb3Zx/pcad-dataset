#!/usr/bin/env python3

import blenderproc as bproc
import sys
bproc.init()
import os

# hack-around because blenderproc removes modules loaded with PYTHONPATH
sys.path.insert(0, os.environ['PWD'])
sys.path.insert(0, "/opt/conda/envs/blender/lib/python3.10/site-packages/")
import pprint

print(pprint.pformat(sys.path))

from PIL import Image
from src.utils import convert_deg_to_rad, image_overlay, filename_property, frame_size
import bpy
import itertools
import mathutils
import numpy as np
import os
import seaborn as sns
import cv2

class DefectMask():
    stl_mappings = {"article-photos-defect-top-left": "top_left",
                    "article-photos-defect-top-right": "top_right",
                    "article-photos-defect-bottom-left": "bottom_left",
                    "article-photos-defect-bottom-right": "bottom_right",
                    "article-photos-defect-front-face": "front_face"}

    NUM_COLORS_DEBUGGING = 100

    def __init__(self, results_root, datasets_root, image_set, _type, location, debugging=False, use_binary_colors=True):
        self.results_root = results_root
        self.datasets_root = datasets_root
        self.image_set = image_set
        self._type = _type
        self.location = location

        self.mask_connector = "2700642_connector_" + self.location

        # bpy.data.objects['2700642_connector_top_left']['category_id'] = 2
        self.base_name = "2700642"
        self.combinations = ['_'.join(pair) for pair in itertools.product(['top', 'bottom'],
                                                                          ['left', 'right'])]
        self.debugging = debugging
        self.use_binary_colors = use_binary_colors

        if self.debugging:
            n = DefectMask.NUM_COLORS_DEBUGGING
        else:
            n = 4
        if self.use_binary_colors:
            self.colors = dict(zip(range(1, n + 1), (255, 255, 255) * n))
            self.colors[0] = (0, 0, 0)
        else:
            palette = sns.color_palette(None, n)
            palette = [[val * 255 for val in entry] for entry in palette]
            self.colors = dict(zip(range(1, n + 1), palette))
            self.colors[0] = (128, 128, 128)


    def defect_mask_perspective(self):
        # only render on frame
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 0

        # assign new camera
        cam = bpy.context.scene.objects['Camera']
        bpy.context.scene.camera = cam

        # blenderproc needs camera to be set explicitly
        cam_pose = bproc.math.build_transformation_mat(cam.location, cam.rotation_euler)
        bproc.camera.add_camera_pose(cam_pose)

        # rendering options
        frame_id = filename_property(scene_fh, "frame", delimiter="_")
        fh_out = f"{self.results_root}/roi_mask/{image_set}/scene_frame={frame_id}.png"
        frame_height, frame_width = frame_size(image_set)
        bpy.context.scene.render.resolution_x = frame_width
        bpy.context.scene.render.resolution_y = frame_height

        # render segmap and save to png directly from data
        data = bproc.renderer.render_segmap()

        # compose segmap
        class_ids = np.unique(data['class_segmaps'][0])
        segmap = data['class_segmaps'][0]
        a = np.zeros((frame_height, frame_width, 3)).astype(np.uint8)
        for class_id in class_ids:
            segmap_mask = (segmap == class_id)
            if class_id != 0:
                if self.debugging:
                    assert class_id in self.colors
                    c = self.colors[class_id]
                else:
                    c = self.colors[(class_id - 1) % 4 + 1]
            else:
                c = self.colors[0]
            a[segmap_mask, :] = c

        contours, _ = cv2.findContours(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(a, contours, -1, (255, 255, 255), cv2.FILLED)

        (Image.fromarray(a)).save(fh_out)

    def detach_from_parent(self, child):
        matrixcopy = child.matrix_world.copy()
        child.parent = None
        child.matrix_world = matrixcopy


    def load_scene(self, scene_fh):
        while len(bpy.data.objects) != 0:
            bpy.data.objects.remove(bpy.data.objects[-1], do_unlink=True)

        # import precomputed positions
        bpy.ops.import_scene.fbx(filepath=os.path.join(f"./{self.results_root}/scene/{image_set}/", scene_fh))

        obj = bpy.data.objects['2700642_front_face']
        self.detach_from_parent(obj)

        # only have one object. Set to any category id != 0
        bpy.data.objects['2700642_front_face']['category_id'] = 1

        for combination in self.combinations:
            obj_name = '_'.join([self.base_name, 'connector', combination])
            obj = bpy.data.objects[obj_name]
            self.detach_from_parent(obj)
            obj['category_id'] = 1

        obj = bpy.data.objects['2700642_body']
        vec = mathutils.Vector((0, -0.001, 0))
        inv = obj.matrix_world.copy()
        inv.invert()
        obj.location = obj.location + (vec @ inv)
        bpy.data.objects['2700642_body']['category_id'] = 0
        bpy.data.objects.remove(bpy.data.objects['2700642'])
        # bpy.data.objects.remove(bpy.data.objects['2700642_body'])


        import pprint
        print(pprint.pformat([repr(obj) for obj in bpy.context.scene.objects]))

if __name__ == "__main__":

    # arguments can also be passed to blenderproc
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str,
                        help="root directory of datasets")
    parser.add_argument("--dataset_name", type=str,
                        help="name of dataset")
    parser.add_argument("--results_root", type=str,
                        help="root directory of results")
    parser.add_argument("--type", type=str,
                        help="type")
    parser.add_argument("--location", type=str,
                        help="location")


    args = parser.parse_args()
    datasets_root = args.datasets_root
    image_set = args.dataset_name
    results_root = args.results_root
    _type = args.type
    location = args.location

    os.makedirs(f"{results_root}/roi_mask/{image_set}", exist_ok=True)

    if _type == 'no-defect':
        exit()

    defect_mask_renderer = DefectMask(results_root, datasets_root, image_set, _type, location, debugging=False, use_binary_colors=True)

    for i, scene_fh in enumerate(sorted(os.listdir(f"./results/scene/{image_set}"))):
        # for obj in bpy.data.objects:
        #     print(obj)

        defect_mask_renderer.load_scene(scene_fh)
        defect_mask_renderer.defect_mask_perspective()
