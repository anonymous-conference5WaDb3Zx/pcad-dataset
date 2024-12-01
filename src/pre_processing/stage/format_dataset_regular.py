#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import tqdm
import yaml
from PIL import Image
import numpy as np
import itertools
import cv2
import argparse
import bpy
import argparse
import logging
import portalocker

from src.utils import create_filename_pack, filename_property, create_filename_pack, blender_center_crop

dataset_name = "own"#
data_base_dir = "data"
results_base_dir = "results"

parser = argparse.ArgumentParser()
parser.add_argument("--datasets-root", type=str,
                    help="root directory of datasets")
parser.add_argument("--dataset-name", type=str,
                    help="name of dataset")
parser.add_argument('--product-name', dest='product_name', type=str)

args = parser.parse_args()
datasets_root = args.datasets_root
image_set = args.dataset_name
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


fn = f'{datasets_root}/products/{product_name}/compiled_scene_0.blend'

bpy.ops.wm.open_mainfile(filepath=fn)

pos_left_upper, pos_right_lower = blender_center_crop(bpy.context.scene)

with open("data/datasets.yaml", 'r') as fh:
    datasets_config = yaml.safe_load(fh)

print(datasets_config)
train_datasets = {}
if datasets_config['train'] != None and any([dataset['name'] == 'good' for dataset in datasets_config['train'].values()]):
    train_datasets['good'] = [('train', dataset) for dataset in datasets_config['train'].values()]

test_datasets = {}
if datasets_config['test'] != None and any([dataset['name'] == 'good' for dataset in datasets_config['test'].values()]):
    test_datasets['good'] = [('test', datasets_config['test']['good'])]

test_datasets['defect'] = []
if datasets_config['test'] != None:
   for key, val in filter(lambda x: 'defect' in x[0], datasets_config['test'].items()):
       test_datasets['defect'].append(('test', val))

dataset_out_dir = Path(results_base_dir, dataset_name)
dataset_out_dir_masked = Path(results_base_dir, '_'.join([dataset_name, "masked"]))

def crop_roi(img: Image, roi_mask: Image, mask_color=(0, 0, 0)):
    img_empty = Image.new("RGB", img.size, mask_color)
    return Image.composite(img, img_empty, roi_mask)

def crop_and_blur(img: Image, img_mask: Image, contour_thickness = 5, kernel_size = (15,15), black_background=False, background_color=(0, 0, 0)):
    #create image crops
    image_black = np.array(crop_roi(img, img_mask.convert('L'), mask_color=(0, 0, 0)))
    image_white = np.array(crop_roi(img, img_mask.convert('L'), mask_color=background_color))

    #blur image
    if black_background:
        blurred_img = cv2.GaussianBlur(image_black, kernel_size, 0)
    else:
        blurred_img = cv2.GaussianBlur(image_white, kernel_size, 0)

    #create black mask
    mask = np.zeros(image_black.shape, np.uint8)

    #convert color space (on image with black background)
    gray = cv2.cvtColor(image_black, cv2.COLOR_BGR2GRAY)

    #thresholding
    (rv, thresh) = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY)#(gray, 60, 255, cv2.THRESH_BINARY)

    # find contours on threshold image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #draw contour on mask
    cv2.drawContours(mask, contours, -1, (255,255,255), contour_thickness)

    #generate output: where mask is white use blurred image, otherwise use image
    if black_background:
        output = np.where(mask==np.array([255, 255, 255]), blurred_img, image_black)
    else:
        output = np.where(mask==np.array([255, 255, 255]), blurred_img, image_white)
    return Image.fromarray(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage format-dataset_regular.')
    parser.add_argument('--dataset-name', dest='dataset_name', type=str, help="Name of dataset")
    parser.add_argument('--datasets-root', dest='datasets_root', type=str, help="Root directory in which dataset is located")
    parser.add_argument('--dataset-type', dest='dataset_type', type=str, help="Type")
    parser.add_argument('--product-name', dest='product_name', type=str, help="Type")

    args = parser.parse_args()

    image_set = args.dataset_name
    datasets_root = Path(args.datasets_root)
    dataset_type = args.dataset_type
    product_name = args.product_name

    chain = []
    if "good" in train_datasets or train_datasets['defect'] != []:
        for category, train_image_sets in train_datasets.items():
            hits = []
            for dataset_type, train_image_set in train_image_sets:
                if train_image_set['location'] == image_set:
                    hits.append((dataset_type, train_image_set))
            chain.append((category, hits))
    print(chain)
    for category, train_image_sets in chain:
        print(category)
        print(train_image_sets)
        for dataset_type, train_image_set in tqdm.tqdm(train_image_sets, desc="Train category"):
            print(train_image_set)
            train_data_dir = Path(data_base_dir, "images", train_image_set['location'])
            train_data_dir_meta = Path(data_base_dir, "images", train_image_set['location'], "pose_est_extra")

            train_imgs = list(filter(lambda x: os.path.isfile(Path(train_data_dir, x)) and "frame-snapshot" in x,
                                  os.listdir(train_data_dir)))
            print(train_data_dir_meta)
            train_imgs_roi = list(filter(lambda x: os.path.isfile(Path(train_data_dir_meta, x)) and "roi_mask" in x,
                                    os.listdir(train_data_dir_meta)))
            print(train_imgs_roi)
            d = Path(results_base_dir, "roi_mask", train_image_set['location'])
            # train_imgs_roi_mask = list(filter(lambda x: os.path.isfile(Path(d, x)),
            #                                   os.listdir(d)))
            #


            fn_pack = create_filename_pack([train_imgs, train_imgs_roi, # train_imgs_roi_mask
                                            ], keyword="frame", delimiter="=")

            for i, (train_img, train_img_mask# , train_img_roi_mask
                    ) in tqdm.tqdm(enumerate(fn_pack),
                                                                               desc="Generate train images…",
                                                                               total=len(train_imgs)):
                # if i % 999 != 0:
                #     continue

                img = Image.open(Path(train_data_dir, train_img))
                img_mask = Image.open(Path(train_data_dir_meta, train_img_mask))
                # img_mask_roi = Image.open(Path(d, train_img_roi_mask))

                img = img.crop((*pos_left_upper, *pos_right_lower))
                img_mask = img_mask.crop((*pos_left_upper, *pos_right_lower))
                # img_mask_roi = img_mask_roi.crop((*pos_left_upper, *pos_right_lower))

                size_final = 800
                img = img.resize((size_final, size_final), Image.Resampling.LANCZOS)
                img_mask = img_mask.resize((size_final, size_final), Image.Resampling.NEAREST)

                # img_mask_roi = img_mask_roi.resize((size_final, size_final), Image.Resampling.NEAREST)

                # img_cropped = crop_and_blur(img, img_mask.convert('L'))
                img_cropped = crop_roi(img, img_mask.convert('L'))
                train_dir = Path(dataset_out_dir, product_name, "train", "good")
                train_dir.mkdir(parents=True, exist_ok=True)
                print(Path(train_dir))

                frame_id = int(filename_property(os.path.join(train_data_dir, train_img),
                                                 "frame"))

                img_cropped.save(Path(train_dir, f"{frame_id:03d}.png"))

                # dataset_out_dir_masked = Path(results_base_dir, '_'.join([dataset_name, "masked"]))
                # img_cropped = crop_roi(img_cropped, img_mask_roi.convert('L'))
                # train_dir = Path(dataset_out_dir_masked, "train", "good")
                # train_dir.mkdir(parents=True, exist_ok=True)
                # img_cropped.save(Path(train_dir, f"{i:03d}.png"))
    chain = []

    if "good" in test_datasets or test_datasets['defect'] != []:
        for category, test_image_sets in test_datasets.items():

            hits = []
            for dataset_type, test_image_set in test_image_sets:
                if test_image_set['location'] == image_set:
                    hits.append((dataset_type, test_image_set))
            chain.append((category, hits))
    print("CHAIN")
    print(chain)
    for category, test_image_sets in chain:
        for _i, (dataset_type, test_image_set) in tqdm.tqdm(enumerate(test_image_sets), desc="Test category"):
            test_data_dir = Path(data_base_dir, "images", test_image_set['location'])
            test_data_dir_meta = Path(data_base_dir, "images", test_image_set['location'], "pose_est_extra")

            test_imgs = list(filter(lambda x: os.path.isfile(Path(test_data_dir, x)) and "frame-snapshot" in x,
                                    os.listdir(test_data_dir)))
            test_imgs_roi = list(filter(lambda x: os.path.isfile(Path(test_data_dir_meta, x)) and "roi_mask" in x,
                                   os.listdir(test_data_dir_meta)))

            if "type" in test_image_set:
                if test_image_set["type"] == "texture":
                    dir_masks = str(Path(results_base_dir, "defect_mask_texture", test_image_set['location']))
                else:
                    dir_masks = str(Path(results_base_dir, "defect_mask", test_image_set['location']))
            else:
                dir_masks = str(Path(results_base_dir, "defect_mask", test_image_set['location']))


            defect_mask_imgs = [str(Path(dir_masks, fn)) for fn in filter(lambda x: os.path.isfile(Path(dir_masks, x)) and not x.endswith('_ortho'),
                                                                  os.listdir(dir_masks))]


            d = Path(results_base_dir, "roi_mask", test_image_set['location'])

            # test_imgs_roi_mask = list(filter(lambda x: os.path.isfile(Path(d, x)),
            #                                  os.listdir(d)))

            # print(test_imgs_roi_mask)

            fn_pack = create_filename_pack([test_imgs, test_imgs_roi, defect_mask_imgs# test_imgs_roi_mask
                                            ],
                                           keyword="frame",
                                           delimiter="=")

            for i, (test_img, test_img_mask, img_defect_mask# , test_img_roi_mask
                    ) in tqdm.tqdm(enumerate(fn_pack),
                                                                             desc="Generate test images…",
                                                                             total=len(test_imgs)):
                # if i % 999 != 0:
                #     continue
                frame_id = int(filename_property(os.path.join(test_data_dir, test_img),
                                                 "frame"))

                size_final = 800
                img_defect_mask = Image.open(img_defect_mask)
                img_defect_mask = img_defect_mask.crop((*pos_left_upper, *pos_right_lower))
                img_defect_mask = img_defect_mask.resize((size_final, size_final), Image.Resampling.NEAREST)

                # check if all values are 0 (black)
                if np.all(np.array(img_defect_mask) == 0):
                    logging.debug("ALL BLACK SKIP")
                    continue

                img = Image.open(Path(test_data_dir, test_img))
                img_mask = Image.open(Path(test_data_dir_meta, test_img_mask))
                # img_mask_roi = Image.open(Path(d, test_img_roi_mask))

                img = img.crop((*pos_left_upper, *pos_right_lower))
                img_mask = img_mask.crop((*pos_left_upper, *pos_right_lower))
                # img_mask_roi = img_mask_roi.crop((*pos_left_upper, *pos_right_lower))

                img = img.resize((size_final, size_final), Image.Resampling.LANCZOS)
                img_mask = img_mask.resize((size_final, size_final), Image.Resampling.NEAREST)
                # img_mask_roi = img_mask_roi.resize((size_final, size_final), Image.Resampling.NEAREST)
                #




                # img_cropped = crop_and_blur(img, img_mask.convert('L'), background_color=(253, 254, 255), contour_thickness=30, kernel_size=(5, 5))
                img_cropped = crop_roi(img, img_mask.convert('L'))

                test_dir = Path(dataset_out_dir, product_name, "test", test_image_set['location'].split("/")[1])
                test_dir.mkdir(parents=True, exist_ok=True)

                print(Path(test_dir, f"{frame_id:03d}.png"))


                img_cropped.save(Path(test_dir, f"{frame_id:03d}.png"))

                # img_cropped = crop_roi(img_cropped, img_mask_roi.convert('L'))
                # test_dir = Path(dataset_out_dir_masked, dataset_type, test_image_set['name'])
                # test_dir.mkdir(parents=True, exist_ok=True)
                # img_cropped.save(Path(test_dir, f"{i:03d}.png"))


                # if category == 'good':
                #     # generate empty masks for good
                #     img_empty = Image.new('RGB', (size_final, size_final), color='black')

                #     if dataset_type == 'test':
                #         gt_dir = Path(dataset_out_dir, "ground_truth", image_set, test_image_set['name'])
                #     else:
                #         gt_dir = Path(dataset_out_dir, "val_ground_truth", image_set, test_image_set['name'])

                #     gt_dir.mkdir(parents=True, exist_ok=True)
                #     img_empty.save(Path(gt_dir, f"{i:03d}_mask.png"))

                #     if dataset_type == 'test':
                #         gt_dir = Path(dataset_out_dir_masked, "ground_truth", image_set, test_image_set['name'])
                #     else:
                #         gt_dir = Path(dataset_out_dir_masked, "val_ground_truth", image_set, test_image_set['name'])

                #     gt_dir.mkdir(parents=True, exist_ok=True)
                #     img_empty.save(Path(gt_dir, f"{i:03d}_mask.png"))

            if category == 'good':
                continue


            if "type" in test_image_set:
                if test_image_set["type"] == "texture":
                    dir_masks = str(Path(results_base_dir, "defect_mask_texture", test_image_set['location']))
                else:
                    dir_masks = str(Path(results_base_dir, "defect_mask", test_image_set['location']))
            else:
                dir_masks = str(Path(results_base_dir, "defect_mask", test_image_set['location']))


            mask_imgs = [str(Path(dir_masks, fn)) for fn in filter(lambda x: os.path.isfile(Path(dir_masks, x)) and not x.endswith('_ortho'),
                                                                  os.listdir(dir_masks))]
            print(dir_masks)
            print(mask_imgs)

            fn_pack = create_filename_pack([test_imgs, mask_imgs, # test_imgs_roi_mask
                                            ],
                                           keyword="frame",
                                           delimiter="=")


            for i, (test_img, mask_img) in tqdm.tqdm(enumerate(fn_pack),
                                                     desc="Generate masks…",
                                                     total=len(test_imgs)):

                frame_id = filename_property(test_img, 'frame')

                print(test_img)
                print(mask_img)

                print(test_img)
                img = Image.open(mask_img)
                img = img.crop((*pos_left_upper, *pos_right_lower))
                size_final = 800
                img = img.resize((size_final, size_final), Image.Resampling.NEAREST)

                # check if all values are 0 (black)
                if np.all(np.array(img) == 0):
                    print("ALL BLACK")
                    continue

                # if dataset_type == 'test':
                gt_dir = Path(dataset_out_dir, product_name, "ground_truth", test_image_set['location'].split("/")[1])
                # else:
                #     gt_dir = Path(dataset_out_dir, "val_ground_truth", image_set)

                gt_dir.mkdir(parents=True, exist_ok=True)
                img_rgb = Image.new("L", img.size)
                img_rgb.paste(img, (0, 0))

                img_rgb.save(Path(gt_dir, f"{frame_id:03d}.png"))
                # dataset_out_dir_masked = Path(results_base_dir, '_'.join([dataset_name, "masked"]))
                # if dataset_type == 'test':
                # gt_dir = Path(dataset_out_dir_masked, test_image_set['location'].split("/")[0],
                #                   "ground_truth", test_image_set['location'].split("/")[1])
                # # else:
                # #     gt_dir = Path(dataset_out_dir_masked, "val_ground_truth", image_set)
                # gt_dir.mkdir(parents=True, exist_ok=True)
                # img_rgb.save(Path(gt_dir, f"{frame_id:03d}_mask.png"))
