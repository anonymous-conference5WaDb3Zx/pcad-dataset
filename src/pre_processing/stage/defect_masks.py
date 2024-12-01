#!/usr/bin/env python3

import blenderproc as bproc
import sys
bproc.init()
import os

# hack-around because blenderproc removes modules loaded with PYTHONPATH
sys.path.insert(0, os.environ['PWD'])
sys.path.insert(0, "/opt/conda/envs/blender/lib/python3.10/site-packages/")
sys.path.insert(0, "/home/mambauser/.local/lib/python3.10/site-packages/")
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
import logging
import portalocker

class DefectMask():
    NUM_COLORS_DEBUGGING = 100

    def __init__(self, results_root, datasets_root, image_set, _type, location, product_name, debugging=False, use_binary_colors=True):
        self.results_root = results_root
        self.datasets_root = datasets_root
        self.image_set = image_set
        self._type = _type
        self.location = location
        self.product_name = product_name


        # bpy.data.objects['2700642_connector_top_left']['category_id'] = 2
        self.base_name = self.product_name
        # self.combinations = ['_'.join(pair) for pair in itertools.product(['top', 'bottom'],
        #                                                                   ['left', 'right'])]

        import yaml
        with open(os.path.join(self.datasets_root, "products", product_name, "meta.yaml"), 'r') as fh:
            self.meta = yaml.safe_load(fh)

        self.mask_connector = f"{product_name}_" + self.location

        self.combinations = self.meta['components']

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


    def defect_mask_front_ortho(self):
        # only render on frame
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 0

        cam = bpy.context.scene.objects['Camera']
        bpy.context.scene.camera = cam
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = 0.15

        # Set the camera to be orthographic and set the orthographic scale
        # Location has no effect on projection of image for orthographic camera.
        location = (0, 1, 0)
        rotation_euler = convert_deg_to_rad((-90, 0, 0))
        cam_pose = bproc.math.build_transformation_mat(location, rotation_euler)
        bproc.camera.add_camera_pose(cam_pose)

        fh_out = f"{self.results_root}/defect_mask/{image_set}/scene_frame_ortho.png"
        # frame_height, frame_width = frame_size()
        frame_height = frame_width = 900
        bpy.context.scene.render.resolution_x = frame_width
        bpy.context.scene.render.resolution_y = frame_height

        if self.debugging:
            bpy.data.objects['2700642_body']['category_id'] = 1
        else:
            bpy.data.objects['2700642_body']['category_id'] = 0

        for i, combination in enumerate(self.combinations):
            obj_name = '_'.join([self.base_name, combination])
            if obj_name in bpy.data.objects:
                if obj_name == self.mask_connector:
                    cat = 1
                else:
                    if self.debugging:
                        # offset by to to get out of range [0, 1] for vibrant debugging colors
                        cat = i + 2
                    else:
                        cat = 0
                print(f"object {obj_name} gets {cat}.")
                bpy.data.objects[obj_name]['category_id'] = cat

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
        (Image.fromarray(a)).save(fh_out)


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
        fh_out = f"{self.results_root}/defect_mask/{image_set}/scene_frame={frame_id}.png"
        frame_height, frame_width = frame_size(self.image_set)
        bpy.context.scene.render.resolution_x = frame_width
        bpy.context.scene.render.resolution_y = frame_height

        # for i, combination in enumerate(self.combinations):
        #     obj_name = '_'.join([self.base_name, combination])
        #     if obj_name in bpy.data.objects:
        #         if obj_name == self.mask_connector:
        #             cat = 1
        #         else:
        #             if self.debugging:
        #                 # offset by to to get out of range [0, 1] for vibrant debugging colors
        #                 cat = i + 2
        #             else:
        #                 cat = 0
        #         print(f"object {obj_name} gets {cat}.")
        #         bpy.data.objects[obj_name]['category_id'] = cat



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
        (Image.fromarray(a)).save(fh_out)

    def construct_default_scene(self, product_name):
        while len(bpy.data.objects) != 0:
            bpy.data.objects.remove(bpy.data.objects[-1], do_unlink=True)

        # create new camera
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)

        for obj in bpy.context.scene.objects:
            print(obj)

        for combination in self.combinations:
            fnh = '_'.join([product_name, combination]) + '.stl'
            fn = os.path.join(self.datasets_root, "products", product_name, fnh)
            bpy.ops.import_mesh.stl(filepath=fn, global_scale=0.001)

        # load stl for body
        fnh = '_'.join([product_name, 'body']) + '.stl'
        fn = os.path.join(self.datasets_root, "products", product_name, fnh)
        bpy.ops.import_mesh.stl(filepath=fn, global_scale=0.001)


        target = bpy.context.scene.objects['2700642_body']

        # link additional parts to body
        for combination in self.combinations:
            obj_name = '_'.join([self.base_name, combination])
            bpy.context.scene.objects[obj_name].parent = target

        import yaml
        with open(os.path.join(self.datasets_root, "products", product_name, "meta.yaml"), 'r') as fh:
            meta = yaml.safe_load(fh)

        target.rotation_euler = mathutils.Vector(convert_deg_to_rad(meta['r_euler']))
        target.location = mathutils.Vector(meta['displacement'])
        # target.rotation_euler = mathutils.Vector(convert_deg_to_rad((0, 90, 0)))
        # target.location = mathutils.Vector((-119.82+60.35, 0, 0))/1000
        bpy.ops.object.transform_apply(rotation=True)



    def load_scene(self, scene_fh, product_name, subtype):
        while len(bpy.data.objects) != 0:
            bpy.data.objects.remove(bpy.data.objects[-1], do_unlink=True)

        # import precomputed positions
        bpy.ops.import_scene.fbx(filepath=os.path.join(f"./{self.results_root}/scene/{image_set}/", scene_fh))
        for k, v in bpy.data.objects.items():
            if f"{product_name}_body" not in k and all([comp not in k for comp in self.combinations]) and "Camera" not in k:
                bpy.data.objects.remove(bpy.data.objects[k])

        for obj in bpy.context.selected_objects:
            print(obj)


        # only have one object. Set to any category id != 0
        if self.debugging:
            bpy.data.objects[f'{product_name}_body']['category_id'] = 1
        else:
            bpy.data.objects[f'{product_name}_body']['category_id'] = 0


        for i, combination in enumerate(self.combinations):
            obj_name = '_'.join([self.base_name, combination])
            # print(obj_name)
            # print(self.mask_connector)
            if obj_name in bpy.data.objects:
                if subtype == "missing":
                    if obj_name == self.mask_connector:
                        cat = 1
                    else:
                        if self.debugging:
                            cat = i + 2
                        else:
                            cat = 0
                    print(f"object {obj_name} gets {cat}.")
                    bpy.data.objects[obj_name]['category_id'] = cat
                elif subtype == "missing_exchange":
                    print([t for t in self.meta['exchange'][self.image_set.split('/')[1]]])
                    print(obj_name)
                    if any([t in obj_name for t in self.meta['exchange'][self.image_set.split('/')[1]]]):
                        cat = 1
                    else:
                        if self.debugging:
                            cat = i + 2
                        else:
                            cat = 0
                    print(f"object {obj_name} gets {cat}.")
                    bpy.data.objects[obj_name]['category_id'] = cat


if __name__ == "__main__":
    #
    #
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
    parser.add_argument('--product-name', dest='product_name', type=str, help="Type")
    parser.add_argument('--subtype', dest='subtype', type=str, help="Type")

    args = parser.parse_args()
    datasets_root = args.datasets_root
    image_set = args.dataset_name
    results_root = args.results_root
    product_name = args.product_name
    _type = args.type
    location = args.location
    subtype = args.subtype


    os.makedirs(f"{results_root}/defect_mask/{image_set}", exist_ok=True)

    stage_name = os.path.splitext(os.path.basename(__file__))[0]
    logfnh = f"{stage_name}@{image_set}".replace("/", "-") + ".log"
    logfn = os.path.join("logs", logfnh)
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
    level=logging.DEBUG,
    format=f"[Stage={stage_name}, image_set={image_set}] %(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logfn),
        logging.StreamHandler()])

    print = logging.debug

    lock_fn = "lock.lock"

    with open(lock_fn, 'w') as fh:
        portalocker.lock(fh, portalocker.LOCK_EX)
        fh.writelines(['unlocked'])
        logging.debug("*" * 100)
        logging.debug("UNLOCKED")
        portalocker.unlock(fh)


    if _type == 'no-defect':
        exit()

    defect_mask_renderer = DefectMask(results_root, datasets_root, image_set, _type, location, product_name, debugging=False, use_binary_colors=True)

    for i, scene_fh in enumerate(filter(lambda x: "fbx" in x, sorted(os.listdir(f"./results/scene/{image_set}")))):
        if subtype != "missing" and subtype != "missing_exchange":
            continue

        defect_mask_renderer.load_scene(scene_fh, product_name, subtype)
        defect_mask_renderer.defect_mask_perspective()
