#!/usr/bin/env python3

from PIL import Image
from mathutils import Vector, Matrix, Euler
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from typing import Optional
from src.utils import convert_deg_to_rad, cv_to_blender, image_overlay, frame_size, filename_property
import bpy
import csv
import glob
import mathutils
import numpy as np
import os
import pickle
import cv2
import yaml
import itertools

import pprint
import argparse
from pathlib import Path

import portalocker
import logging

def rt_mat(rot_mat, tvec):
    mat = np.zeros((1, 4, 4))
    mat[:, :3, :3] = rot_mat
    mat[:, 3, 3] = 1
    mat[:, :3, 3] = tvec
    return mat

def rot_mat_to_rt_mat(rot_mat):
    return rt_mat(rot_mat, np.array([0, 0, 0]).T)

def np_mat_to_mat(np_mat):
    # for row in range(mat.shape[-2]):
        # print(mat[..., row, :])
    # print([mat[..., row, :].squeeze().tolist() for row in range(mat.shape[-2])])
    return mathutils.Matrix([np_mat[..., row, :].squeeze().tolist() for row in range(np_mat.shape[-2])])

def mat_to_np_mat(mat):
    np_mat = np.empty((len(mat), len(mat[0])))
    for row in range(len(mat)):
        np_mat[row, :] = mat[row]
    return np_mat


# source: https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
def average_quaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].flatten())

def aruco_marker_pose(aruco_id: int, aruco_sketch_parameters):
    assert aruco_id in [0, 1, 2, 3, 4]
    assert isinstance(aruco_id, int)

    params = aruco_sketch_parameters
    pre = ["aruco", str(aruco_id), "position"]
    x = params["_".join(pre + ["horizontal"])] * 0.001
    z = params["_".join(pre + ["vertical"])] * 0.001

    rot = params["_".join(["aruco", str(aruco_id), "rotation"])]

    # compensate fusion-specific rotation representation in sketches
    if rot > 90:
        rot -= 180

    # finally, invert to ahere to blender
    rot *= -1

    return {'pos': mathutils.Vector((x, z, 0)), 'rot': mathutils.Vector(convert_deg_to_rad((0, 0, rot)))}

def article_position(aruco_sketch_parameters):
    params = aruco_sketch_parameters
    pre = ["article", "position"]
    x = params["_".join(pre + ["horizontal"])] * 0.001
    z = params["_".join(pre + ["vertical"])] * 0.001
    return {'pos': mathutils.Vector((x, z, 0))}

def read_aruco_scene_sketch_parameters() -> dict:
    with open("./data/aruco_scene_parameters.csv", "r") as fh:
        sketch_parameters = {}
        lines = csv.reader(fh, delimiter=',')
        for line in lines:
            k = line[0]
            unit = line[1]
            value_raw = line[2]

            # is referenced
            if unit not in value_raw:
                print(f"Skip referenced value for k={k}")
                continue

            # regular value
            value = line[2][:-(len(unit) + 1)]
            sketch_parameters[line[0]] = float(value)
    return sketch_parameters


def positional_offset(aruco_id: int):
    # we have four aruco markers on the scene get positional offset to position marker

    with open("./data/aruco_scene_parameters.csv", "r") as fh:
        lines = csv.reader(fh, delimiter=',')


def init_scene(product_name, image_set):
    # clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # load scene from template
    bpy.ops.import_scene.fbx(filepath="./data/template_scene2.fbx")        # Linux

    h, w = frame_size(image_set)

    # TODO: mtx is the same for all images. Make global
    mtx = np.load(f"data/images/2700642-training/no-defect/pose_est_extra/mtx_frame=0.npy")

    # fix sensor size to arbitrary value
    sensor_size = (10.5, 5.4)

    # set scene camera
    cam = bpy.context.scene.objects['Camera']
    bpy.context.scene.camera = cam
    bpy.data.cameras['Camera'].lens = mtx[0][0] / w * sensor_size[0]
    bpy.data.cameras['Camera'].sensor_fit = "AUTO"
    bpy.data.cameras['Camera'].sensor_width = sensor_size[0]
    bpy.data.cameras['Camera'].sensor_height = sensor_size[1]
    frame_height, frame_width = frame_size(image_set)
    bpy.context.scene.render.resolution_x = frame_width
    bpy.context.scene.render.resolution_y = frame_height

    # shift of principal point
    bpy.data.cameras['Camera'].shift_x = -(mtx[0][2] / w - 0.5)
    bpy.data.cameras['Camera'].shift_y = (mtx[1][2] - 0.5 * h) / w
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = mtx[1][1] / mtx[0][0]

    for k, v in bpy.data.cameras['Camera'].items():
        print(k, v)

    for obj in bpy.context.selected_objects:
        print(obj)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['arucomarker_platte v1'].select_set(True)
    bpy.ops.object.delete()
    scaling_aruco = 2
    scaling = 0.001 * scaling_aruco

    data_dir = "./data"
    import yaml
    with open(os.path.join(data_dir, "products", product_name, "meta.yaml"), 'r') as fh:
        meta = yaml.safe_load(fh)

    if 'scale' in meta:
        scaling_import_stl = meta['scale']
    else:
        scaling_import_stl = 0.001

    combinations = meta['components']


    # bpy.ops.import_mesh.stl(filepath="./data/arucomarker_platte v1.stl", global_scale=scaling)

    # BUG: for unknown reason, the order of importing matters.
    # Connectors first, body second. The reference coordinate systems seems to shift over to the new part.
    #
    # load stl for body with connectors: needed for uv mapping
    # fnh = product_name + '.stl'
    # fn = os.path.join(data_dir, "products", product_name, fnh)
    # bpy.ops.import_mesh.stl(filepath=fn, global_scale=0.001)
    # obj = bpy.data.objects[product_name]

    # bpy.context.scene.collection.objects.unlink(obj)
    #
    # load stl for additional parts
    for combination in combinations:
        fnh = '_'.join([product_name, combination]) + '.stl'
        fn = os.path.join(data_dir, "products", product_name, fnh)
        bpy.ops.import_mesh.stl(filepath=fn, global_scale=scaling_import_stl)


    # load stl for body
    fnh = '_'.join([product_name, 'body']) + '.stl'
    fn = os.path.join(data_dir, "products", product_name, fnh)
    bpy.ops.import_mesh.stl(filepath=fn, global_scale=scaling_import_stl)

    # # load stl for body
    fnh = '_'.join([product_name, 'front_face']) + '.stl'
    if os.path.exists(fnh):
        fn = os.path.join(data_dir, "products", product_name, fnh)
        bpy.ops.import_mesh.stl(filepath=fn, global_scale=scaling_import_stl)

    target = bpy.context.scene.objects[f'{product_name}_body']


    # link additional parts to body
    for combination in combinations:
        obj_name = '_'.join([product_name, combination])
        bpy.context.scene.objects[obj_name].parent = target


    target.rotation_euler = mathutils.Vector(convert_deg_to_rad(meta['r_euler']))
    target.location = mathutils.Vector(meta['displacement'])

    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.ops.object.transform_apply(rotation=True)

    return cam, target

def render_scene(fh_out):
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(fh_out)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    # bpy.context.scene.render.resolution_x = 960
    # bpy.context.scene.render.resolution_y = 540
    bpy.ops.render.render(use_viewport=True, write_still=True)

    # img = Image.open(fh_out)
    # pixdata = img.load()
    # for x in range(img.size[0]):
    #     for y in range(img.size[1]):
    #         if pixdata[x, y][:3] == (0, 0, 0):
    #             pixdata[x, y] = (0, 0, 0, 0)

    # img.save(fh_out, "PNG")

# save current view mode
# mode = bpy.context.scene.area.type

# set view mode to 3D to have all needed variables available
# bpy.context.scene.area.type = "VIEW_3D"
#

def aruco_tvecs(frame_id):
    marker_tvec_fns: list = sorted(glob.glob(f"./data/images/{image_set}/pose_est_extra/tvec_*frame={frame_id}.npy"))
    tvec_aruco_dict = {}

    for marker_tvec_fn in marker_tvec_fns[:]:
        fnh = os.path.basename(marker_tvec_fn).split('/')[-1]
        aruco_id = int(filename_property(fnh, "id"))
        tvec_aruco = np.load(f"./data/images/{image_set}/pose_est_extra/tvec_id={aruco_id}_frame={frame_id}.npy")
        tvec_aruco_dict[aruco_id] = tvec_aruco

    return tvec_aruco_dict


def aruco_rvecs(frame_id):
    rvec_aruco_dict = {}
    # TODO: replace tvec with rvec
    marker_tvec_fns: list = sorted(glob.glob(f"./data/images/{image_set}/pose_est_extra/tvec_*frame={frame_id}.npy"))
    for marker_tvec_fn in marker_tvec_fns[:]:
        fnh = os.path.basename(marker_tvec_fn).split('/')[-1]
        aruco_id = int(filename_property(fnh, "id"))
        rvec_aruco = np.load(f"./data/images/{image_set}/pose_est_extra/rvec_id={aruco_id}_frame={frame_id}.npy")
        rvec_aruco_blender = cv_to_blender(rvec_aruco)

        rvec_aruco_dict[aruco_id] = rvec_aruco_blender

    return rvec_aruco_dict

def center_position_global(frame_id,
                           aruco_id,
                           displacement=mathutils.Vector((0.193 + 0.045 - 0.004 - 0.007, 0, 0.003)),
                           rotation_prefix=np.array([0, 0, 90])):
    tvec_aruco = aruco_tvecs(frame_id)[aruco_id]
    rvec_aruco = aruco_rvecs(frame_id)[aruco_id]

    aruco_rot_mat = np_mat_to_mat(R.from_rotvec(rvec_aruco.squeeze()).as_matrix()) @ np_mat_to_mat(R.from_euler('xyz', -rotation_prefix, degrees=True).as_matrix())
    aruco_rot_mat_np = mat_to_np_mat(aruco_rot_mat)

    aruco_pos = mathutils.Vector(cv_to_blender(tvec_aruco).squeeze())
    plate_mat = mathutils.Matrix(np_mat_to_mat(rt_mat(aruco_rot_mat_np,
                                                      aruco_pos)))

    article_pos = cam.matrix_world.inverted() @ plate_mat @ ((plate_mat.inverted() @ cam.matrix_world @ aruco_pos) + displacement)

    return article_pos

def debug_center_positions(frame_id, num_dominant_aruco_markers=3):
    tvec_aruco_dict = aruco_tvecs(frame_id)

    center_position_global_dict = {}
    for aruco_id in tvec_aruco_dict.keys():
        center_position_global_dict[aruco_id] = center_position_global(frame_id, aruco_id)

    tvec_aruco_length_dict = dict([(k, np.linalg.norm(v)) for k, v in tvec_aruco_dict.items()])
    for j, (d, pos) in enumerate(sorted(zip(dict(sorted(tvec_aruco_length_dict.items())).values(),
                                            dict(sorted(center_position_global_dict.items())).values()))[:num_dominant_aruco_markers]):
        bpy.ops.mesh.primitive_cube_add(size=0.0075)
        center_obj = bpy.context.scene.objects[-1]
        center_obj.scale = (1, 1, 1)
        center_obj.location = pos


def avg_center_position_global(frame_id,
                               num_dominant_aruco_markers=3):

    tvec_aruco_dict = aruco_tvecs(frame_id)
    tvec_aruco_length_dict = dict([(k, np.linalg.norm(v)) for k, v in tvec_aruco_dict.items()])

    center_position_global_dict = {}
    for aruco_id in tvec_aruco_dict.keys():
        center_position_global_dict[aruco_id] = center_position_global(frame_id, aruco_id)

    positions = np.zeros(shape=(min(num_dominant_aruco_markers, len(tvec_aruco_length_dict)), 3))
    for j, (d, pos) in enumerate(sorted(zip(dict(sorted(tvec_aruco_length_dict.items())).values(),
                                            dict(sorted(center_position_global_dict.items())).values()))[:num_dominant_aruco_markers]):
        pos = np.array([pos[0],
                        pos[1],
                        pos[2]])

        positions[j, :] = pos

    position_avg = np.mean(positions, axis=0)
    position_avg = mathutils.Vector((position_avg[0],
                                     position_avg[1],
                                     position_avg[2]))

    return position_avg

def avg_rotation(frame_id,
                 num_dominant_aruco_markers=3):

    tvec_aruco_dict = aruco_tvecs(frame_id)
    tvec_aruco_length_dict = dict([(k, np.linalg.norm(v)) for k, v in tvec_aruco_dict.items()])
    rvec_aruco_dict = aruco_rvecs(frame_id)


    aruco_rotations = {}
    for aruco_id in rvec_aruco_dict.keys():
        aruco_rotations[aruco_id] = np.array([0, 0, (aruco_id * 36)])

    aruco_quaternion = {}
    assert sorted(rvec_aruco_dict.keys()) == sorted(aruco_rotations.keys()), f"{pprint.pformat(sorted(rvec_aruco_dict))},\n{sorted(aruco_rotations)}"
    for (aruco_id, rvec), aruco_rot in zip(sorted(rvec_aruco_dict.items()),
                                           dict(sorted(aruco_rotations.items())).values()):

        aruco_rot_mat = np_mat_to_mat(R.from_rotvec(rvec.squeeze()).as_matrix()) @ np_mat_to_mat(R.from_euler('xyz', -aruco_rot, degrees=True).as_matrix())
        aruco_rot_mat_np = mat_to_np_mat(aruco_rot_mat)

        q = R.from_matrix(aruco_rot_mat_np).as_quat()
        aruco_quaternion[aruco_id] = q

    # avarage rotation from closest 3 aruco markers
    quaternion_mat = np.empty(shape=(num_dominant_aruco_markers, 4))
    for j, (d, q) in enumerate(sorted(zip(dict(sorted(tvec_aruco_length_dict.items())).values(),
                                          dict(sorted(aruco_quaternion.items())).values()))[:3]):
        quaternion_mat[j, :] = q


    return average_quaternions(quaternion_mat)


def aruco_corner_positions(frame_id, aruco_id, aruco_size):
    corner_displacements = [(-aruco_size, -aruco_size, 0),
                            (-aruco_size, +aruco_size, 0),
                            (+aruco_size, -aruco_size, 0),
                            (+aruco_size, +aruco_size, 0),
                            (0, 0, 0)]

    tvec_aruco = aruco_tvecs(frame_id)[aruco_id]
    rvec_aruco = aruco_rvecs(frame_id)[aruco_id]
    for corner_displacement in corner_displacements:
        aruco_pos = mathutils.Vector(cv_to_blender(tvec_aruco).squeeze())
        bpy.ops.mesh.primitive_cube_add(size=0.0075)
        corner_obj = bpy.context.scene.objects[-1]
        corner_obj.scale = (1, 1, 0.2)
        aruco_rot_mat = np_mat_to_mat(R.from_rotvec(rvec_aruco.squeeze()).as_matrix())
        aruco_rot_mat_np = mat_to_np_mat(aruco_rot_mat)
        plate_mat = mathutils.Matrix(np_mat_to_mat(rt_mat(aruco_rot_mat_np,
                                                          aruco_pos)))

        pos = plate_mat @ ((plate_mat.inverted() @ aruco_pos) + mathutils.Vector(corner_displacement))

        corner_obj.location = pos
        corner_obj.rotation_euler = np_mat_to_mat(R.from_rotvec(rvec_aruco.squeeze()).as_matrix()).to_euler()


def raycasting(frame, cam, target):
    topRight = frame[0]
    bottomRight = frame[1]
    bottomLeft = frame[2]
    topLeft = frame[3]

    # number of pixels in X/Y direction
    resolutionX = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
    resolutionY = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))

    # setup vectors to match pixels
    xRange = np.linspace(topLeft[0], topRight[0], resolutionX)
    yRange = np.linspace(topLeft[1], bottomLeft[1], resolutionY)

    # indices for array mapping
    indexX = 0
    indexY = 0

    # df = pd.DataFrame(columns=["values", "normals", "steepness", "location"],
    #                   index=pd.MultiIndex.from_product([range(0, resolutionX + 1),
    #                                                     range(0, resolutionY + 1)]))

    # print(df)

    mwi = cam.matrix_world.inverted()

    # iterate over all targets
    for target in [target]:
        # calculate origin
        matrixWorld = target.matrix_world
        matrixWorldInverted = matrixWorld.inverted()
        origin = matrixWorldInverted @ cam.matrix_world.translation

        # reset indices
        indexX = 0
        indexY = 0

        data = {}
        data['location'] = np.empty((resolutionX, resolutionY, 3))
        data['normals'] = np.empty((resolutionX, resolutionY, 3))
        data['steepness'] = np.empty((resolutionX, resolutionY, 1))

        for k, _ in data.items():
            data[k][:] = np.nan

        # iterate over all X/Y coordinates
        for x in tqdm(xRange):
            for y in yRange:
                # get current pixel vector from camera center to pixel
                pixelVector = Vector((x, y, topLeft[2]))

                # rotate that vector according to camera rotation
                pixelVector.rotate(cam.matrix_world.to_quaternion())

                # calculate direction vector
                destination = matrixWorldInverted @ (pixelVector + cam.matrix_world.translation)
                direction = (destination - origin).normalized()

                # perform the actual ray casting
                hit, location, norm, face =  target.ray_cast(origin, direction)

                if hit:
                    data['location'][indexX, indexY] = (matrixWorld @ location)
                    # conversion in another space is not necessary
                    location_local = mwi @ location
                    location_normal_vector_tip = mwi @ (location + norm)
                    normal_local = location_normal_vector_tip - location_local
                    data['normals'][indexX, indexY] = normal_local
                    orientation_cam = cam.matrix_world @ Vector((0, 0, 1))
                    orientation_cam.normalize()
                    data['steepness'][indexX, indexY] = orientation_cam.dot(norm)
                # update indices
                indexY += 1

            indexX += 1
            indexY = 0

    return data


if __name__ == "__main__":
    # with open("data/datasets.yaml", 'r') as fh:
    #     datasets_config = yaml.safe_load(fh)

    # image_sets = []
    # if "train" in datasets_config:
    #     image_sets += [datasets_config["train"]]
    # if "test" in datasets_config:
    #     if "good" in datasets_config:
    #         image_sets += [datasets_config['test']['good']]
    #     if "detect" in datasets_config:
    #         image_sets += datasets_config['test']['defect']

    parser = argparse.ArgumentParser(description='Stage target-pose')
    parser.add_argument('--dataset-name', dest='dataset_name', type=str, help="Name of dataset")
    parser.add_argument('--datasets-root', dest='datasets_root', type=str, help="Root directory in which dataset is located")
    parser.add_argument('--dataset-type', dest='dataset_type', type=str, help="Type")
    parser.add_argument('--product-name', dest='product_name', type=str, help="Type")

    args = parser.parse_args()

    image_set = args.dataset_name
    datasets_root = Path(args.datasets_root)
    dataset_type = args.dataset_type
    product_name = args.product_name

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


    # import time
    # n = 20
    # for i in range(n):
    #     logging.debug(f"Working step {i}/{n}")
    #     logging.debug(np.random.rand(100, 100, 100).shape)
    #     print("HERE")
    #     time.sleep(1)
    # exit()


    # stage_name = os.path.splitext(os.path.basename(__file__))[0]
    # logfnh = f"{stage_name}@{image_set}".replace("/", "-") + ".log"
    # logfn = os.path.join("logs", logfnh)
    # os.makedirs("logs", exist_ok=True)
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format=f"[Stage {stage_name}] %(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[
    #         logging.FileHandler(logfn),
    #         logging.StreamHandler()])


    # lock_fn = "lock.lock"
    # with open(lock_fn, 'w') as fh:
    #     portalocker.lock(fh, portalocker.LOCK_EX)
    #     fh.writelines(['unlocked'])
    #     portalocker.unlock(fh)

    # with open(lock_fn, 'r') as fh:
    #     portalocker.lock(fh, portalocker.LOCK_EX)
    #     logging.debug("STATE")
    #     logging.debug(fh.readlines())
    #     portalocker.lock(fh, portalocker.LOCK_EX)
    # exit()
    #

    if dataset_type == "defect":
        os.makedirs(f"./results/scene/defect/{image_set}", exist_ok=True)
    else:
        os.makedirs(f"./results/scene/{image_set}", exist_ok=True)

    # get number of images
    number_images = len(list(filter(lambda x: "frame-snapshot" in x, os.listdir(f"./data/images/{image_set}/"))))

    marker_objs = []
    mtx = np.load(f"data/images/2700642-training/no-defect/pose_est_extra/mtx_frame=0.npy")
    frame_height, frame_width = frame_size(image_set)

    import math
    transforms = {'frames': [],
                  'camera_angle_x': 2 * np.arctan2(frame_width, 2 * mtx[0][0]),
                  'camera_angle_x_deg': np.rad2deg(2 * np.arctan2(frame_width, 2 * mtx[0][0])),
                  'camera_angle_x_rad': 2 * np.arctan2(frame_width, 2 * mtx[0][0])}
    transforms_synthetic_rendering = {}

    for i, f in enumerate(sorted(list(filter(lambda x: "frame-snapshot" in x, os.listdir(f"./data/images/{image_set}"))))):
        cam, target = init_scene(product_name, image_set)
        article_pos = avg_center_position_global(i)
        article_rotation = avg_rotation(i)

        rotation_mat = R.from_quat(article_rotation).as_matrix()
        _rt_mat = rt_mat(rotation_mat, article_pos[:])
        _rt_inv = np.linalg.inv(_rt_mat).squeeze()

        frame_id = int(filename_property(f, "frame"))
        transforms['frames'].append({
            'file_path': f"train/good/{frame_id:03d}",
            'transform_matrix': _rt_inv.tolist()
        })

        transforms_synthetic_rendering[i] = {}
        transforms_synthetic_rendering[i]['transform_matrix'] =  _rt_inv.tolist()

        debug_center_positions(i)

        target.location = article_pos
        target.rotation_euler = R.from_quat(article_rotation).as_euler('xyz', degrees=False)


        for aruco_id in aruco_tvecs(i).keys():
            aruco_corner_positions(i, aruco_id, 0.045)

        if dataset_type == "defect":
            bpy.ops.export_scene.fbx(filepath=f"./results/scene/defect/{image_set}/scene_frame_{frame_id}.fbx")
        else:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(),
                                                              f"results/scene/{image_set}/scene_frame_{frame_id}.blend"))
            bpy.ops.export_scene.fbx(filepath=f"./results/scene/{image_set}/scene_frame_{frame_id}.fbx")

    import json
    with open(f"./results/scene/{image_set}/transforms.json", 'w') as fh:
        json.dump(transforms, fh)
    with open(f"./results/scene/{image_set}/transforms_synthetic_rendering.json", 'w') as fh:
        json.dump(transforms_synthetic_rendering, fh)
