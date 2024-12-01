#!/usr/bin/env python3

import blenderproc as bproc
import sys
bproc.init()
import os
# hack-around because blenderproc removes modules loaded with PYTHONPATH
sys.path.insert(0, os.environ['PWD'])
sys.path.insert(0, "/home/rfm/doc/research/dev-container")
# sys.path.insert(0, "/home/rfm/micromamba/envs/segmap-test/lib/python3.10/site-packages/")
sys.path.insert(0, "/opt/conda/envs/blender/lib/python3.10/site-packages/")
sys.path.insert(0, "/home/mambauser/.local/lib/python3.10/site-packages/")

import pprint

print(pprint.pformat(sys.path))


from PIL import Image
from src.utils import convert_deg_to_rad, image_overlay, filename_property, frame_size, blender_center_crop
import bpy
import itertools
import mathutils
import numpy as np
import os
import seaborn as sns
from scipy.spatial.transform import Rotation as R
import logging
import portalocker

def get_entity(group, t, name):
    filters = [lambda x: isinstance(x, t),
               lambda x: x.name == name]
    _group_entities = list(filter(lambda x: all([f(x) for f in filters]),
                               group))
    assert len(_group_entities) != 0, "Not found."
    assert len(_group_entities) == 1, "Must be unique. Found multiple."
    group_entity = _group_entities[0]
    return group_entity

def get_node(material, t, node_name):
    return get_entity(material.node_tree.nodes, t, node_name)

def get_node_input(node, t, name):
    return get_entity(node.inputs, t, name)

def get_node_output(node, t, name):
    return get_entity(node.outputs, t, name)

def get_link(material, node_from, socket_from, node_to, socket_to):
    _links = []
    for link in material.node_tree.links:
        check_list = [link.from_node == node_from,
                      link.from_socket == socket_from,
                      link.to_node == node_to,
                      link.to_socket == socket_to]
        if all(check_list):
            _links.append(link)
    assert len(_links) == 1
    return _links[0]

# Define the filepath of the .blend file to import
#
class FrontTextureOrtho():
    def __init__(self, results_root, datasets_root, image_set, product_name, debugging=False):
        self.results_root = results_root
        self.datasets_root = datasets_root
        self.image_set = image_set
        self.product_name = product_name

        # self.base_name = "2700642"
        self.debugging = debugging
        print(f'{self.results_root}/scene/{self.image_set}/transforms_synthetic_rendering.json')
        import json
        with open(f'{self.results_root}/scene/{self.image_set}/transforms_synthetic_rendering.json') as fh:
            self.transforms = json.load(fh)
            print(self.transforms)

        import yaml
        with open(os.path.join(datasets_root, "products", product_name, "meta.yaml"), 'r') as fh:
            self.meta = yaml.safe_load(fh)

    def load_scene(self, scene_fh):
        while len(bpy.data.objects) != 0:
            bpy.data.objects.remove(bpy.data.objects[-1], do_unlink=True)

        fn = f'{self.datasets_root}/products/{self.product_name}/compiled_scene_0.blend'

        bpy.ops.wm.open_mainfile(filepath=fn)

    def render(self, frame_id):
        print("BEGIN TO RENDER")
        # only render on frame
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 0

        bproc.renderer.set_output_format(enable_transparency=True)
        # obj = bproc.object.create_primitive("MONKEY")

        # Create a point light next to it
        # light = bproc.types.Light()
        # light.set_location([2, -2, 0])
        # light.set_energy(300)

        for obj in bpy.data.objects:
            print(obj)

        # Set the camera to be orthographic and set the orthographic scale
        #
        # load transforms
        #
        mapping_target = bpy.context.scene.objects['object']
        mapping_target.rotation_euler = mathutils.Vector(convert_deg_to_rad(self.meta['r_euler']))
        mapping_target.location = mathutils.Vector(self.meta['displacement'])

        print(frame_id)
        print(self.transforms[str(frame_id)])
        location = np.array(self.transforms[str(frame_id)]['transform_matrix'])[:3, 3]
        rotation_euler = R.from_matrix(np.array(self.transforms[str(frame_id)]['transform_matrix'])[:3, :3]).as_euler('xyz', degrees=False)
        cam_pose = bproc.math.build_transformation_mat(location, rotation_euler)

        bproc.camera.add_camera_pose(cam_pose)

        print("After")
        for obj in bpy.data.objects:
            print(obj)


        os.makedirs(f"{self.results_root}/synthetic_render/{self.image_set}", exist_ok=True)
        fh_out = f"{self.results_root}/synthetic_render/{self.image_set}/render_{frame_id}.png"

        # do rendering
        data = bproc.renderer.render()

        img = Image.fromarray(data['colors'][0])

        scene = bpy.context.scene
        pos_left_upper, pos_right_lower = blender_center_crop(scene)
        print(pos_left_upper)
        print(pos_right_lower)
        img = img.crop((*pos_left_upper, *pos_right_lower))
        size_final = 800
        img = img.resize((size_final, size_final), Image.LANCZOS)
        img.save(fh_out)


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
    parser.add_argument('--product_name', dest='product_name', type=str)
    parser.add_argument("--type", type=str,
                        help="type")


    args = parser.parse_args()
    datasets_root = args.datasets_root
    image_set = args.dataset_name
    results_root = args.results_root
    product_name = args.product_name
    _type = args.type

    defect_mask_renderer = FrontTextureOrtho(results_root, datasets_root, image_set, product_name, debugging=True)

    os.makedirs(f"{results_root}/synthetic_render/{image_set}", exist_ok=True)

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


    if _type != 'no-defect':
        exit()

    for i, scene_fh in enumerate(filter(lambda x: 'blend' in x,
                                        sorted(os.listdir(f"{results_root}/scene/{image_set}")))):
        # scene_fh = None # DEBUGGING
        # i = 0 # DEBUGGING
        frame_id = filename_property(scene_fh, 'frame', delimiter='_')

        defect_mask_renderer.load_scene(scene_fh)
        defect_mask_renderer.render(frame_id)
