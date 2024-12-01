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
from scipy.ndimage import label, sum as ndi_sum

class DefectMaskTexture():
    # stl_mappings = {"article-photos-defect-top-left": "top_left",
    #                 "article-photos-defect-top-right": "top_right",
    #                 "article-photos-defect-bottom-left": "bottom_left",
    #                 "article-photos-defect-bottom-right": "bottom_right",
    #                 "article-photos-defect-front-face": "front_face"}

    NUM_COLORS_DEBUGGING = 100

    def __init__(self, results_root, datasets_root, image_set, _type, subtype, location, product_name, debugging=False, use_binary_colors=True):
        self.results_root = results_root
        self.datasets_root = datasets_root
        self.image_set = image_set
        self._type = _type
        self.location = location
        self.product_name = product_name
        self.subtype = subtype


        # bpy.data.objects['2700642_connector_top_left']['category_id'] = 2
        # self.combinations = ['_'.join(pair) for pair in itertools.product(['top', 'bottom'],
        #                                                                   ['left', 'right'])]

        import yaml
        with open(os.path.join(self.datasets_root, "products", product_name, "meta.yaml"), 'r') as fh:
            meta = yaml.safe_load(fh)

    def remove_small_blobs(self, binary_array, size_filter):
        # Label connected components
        labeled_array, num_features = label(binary_array)

        # Get sizes of connected components
        component_sizes = np.bincount(labeled_array.ravel())

        # Create a mask with components size greater than 10 pixels
        remove_small_components = component_sizes < size_filter
        remove_small_components_mask = remove_small_components[labeled_array]

        # Set pixels of small components to black
        filtered_array = binary_array.copy()
        filtered_array[remove_small_components_mask] = 0
        return filtered_array

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


    def defect_mask_perspective(self):
        # only render on frame
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 0

        bpy.ops.object.select_all(action='SELECT')
        for obj in bpy.context.selected_objects:
            print(obj)


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
        # Find all materials
        #

        print([m.name for m in bpy.data.materials])
        bpy.data.materials.remove(bpy.data.materials['Dots Stroke'])

        for k, v in bpy.data.objects.items():
            bpy.ops.object.select_all(action='DESELECT')
            print(k)
            print(v)
            if "defective-component" in k:
                print("Set texture")
                img = bpy.data.images.load(f"./data/products/{self.product_name}/{k}.png")
                bpy.data.objects[k].select_set(True)
                mat = bpy.context.active_object.material_slots[0].material
                mat.node_tree.nodes["Image Texture"].image = img


        # render segmap and save to png directly from data
        bproc.renderer.set_render_devices(use_only_cpu=False)
        bpy.ops.object.select_all(action='SELECT')

        data = bproc.renderer.render()

        fh_out = f"{self.results_root}/defect_mask_texture/{image_set}/scene_frame={frame_id}.png"
        print(data['colors'][0].shape)
        print(np.min(data['colors'][0]))
        print(np.max(data['colors'][0]))

        img = self.rgb2gray(data['colors'][0])
        img[img > 1] = 255
        img[img <= 1] = 0
        print(np.min(img))
        print(np.max(img))
        print(np.unique(img))
        min_blob_size = 200
        img = self.remove_small_blobs(img, min_blob_size)
        (Image.fromarray(img).convert('L')).save(fh_out)

    def load_scene(self, scene_fh):
        while len(bpy.data.objects) != 0:
            bpy.data.objects.remove(bpy.data.objects[-1], do_unlink=True)

        print(self.image_set.split('/'))
        blend_name = self.image_set.split('/')[1] + '.blend'
        fn = f'{self.datasets_root}/products/{self.product_name}/{blend_name}'

        bpy.ops.wm.open_mainfile(filepath=fn)

        # import precomputed positions
        bpy.ops.import_scene.fbx(filepath=os.path.join(f"./{self.results_root}/scene/{image_set}/", scene_fh))
        # bpy.data.objects.remove(bpy.data.objects['2700642'])
        # bpy.data.objects.remove(bpy.data.objects['2700642_front_face'])

        bpy.ops.object.select_all(action='SELECT')
        for obj in bpy.context.selected_objects:
            print(obj)

        # Select object to
        import yaml
        with open(os.path.join("./data", "products", self.product_name, "meta.yaml"), 'r') as fh:
            meta = yaml.safe_load(fh)


        target = bpy.context.scene.objects[(self.image_set.split('/')[1]).removeprefix("defect-")]
        target.rotation_euler = mathutils.Vector(convert_deg_to_rad(meta['r_euler']))
        target.location = mathutils.Vector(meta['displacement'])
        bpy.ops.object.select_all(action='DESELECT')
        target.select_set(True)
        bpy.ops.object.transform_apply(rotation=True)
        bpy.ops.object.select_all(action='DESELECT')

        source = bpy.context.scene.objects[self.product_name + "_body"]
        target.rotation_euler = source.rotation_euler
        target.location = source.location

        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(),
                                                          f"tmp.blend"))

        # remove everything except camera and object
        for k, v in bpy.data.objects.items():
            if 'Camera' not in str(k) and 'exchange' not in str(k) and 'defective-component' not in str(k):
                bpy.data.objects.remove(v, do_unlink=True)
                print(f"Remove {k}")
            else:
                print(f"Did not remove {k}")



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


    defect_mask_renderer = DefectMaskTexture(results_root, datasets_root, image_set, _type, subtype, location, product_name, debugging=False, use_binary_colors=True)

    for i, scene_fh in enumerate(filter(lambda x: "fbx" in x, sorted(os.listdir(f"./results/scene/{image_set}")))):
        os.makedirs(f"{results_root}/defect_mask_texture/{image_set}", exist_ok=True)
        if _type == 'no-defect':
            exit()


        if subtype != "texture":
            continue

        defect_mask_renderer.load_scene(scene_fh)
        defect_mask_renderer.defect_mask_perspective()
