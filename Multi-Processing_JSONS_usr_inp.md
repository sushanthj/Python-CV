---
layout: page
title: Multi-Processing, JSONS, and User-Inputs
permalink: /Multi-Proc, JSON, usr_inp/
nav_order: 13
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Before you Begin
{: .fs-9 }

[Understand the Difference between Multi-Proc and Multi-Thread](https://www.geeksforgeeks.org/difference-between-multiprocessing-and-multithreading/){: .btn .fs-5 .mb-4 .mb-md-0 }

# Understanding the Pool loop for Multi-Processsing

Example code:

```python
from multiprocessing import Pool
from tqdm import tqdm
#tqdm is used to show the progress bar, it's just a wrapper around a for loop as seen below

#We define a random list
annotations = [{'project_name':'1'}, {'project_name':'2'}]

def print_images_labels(annotation):
    project_name = annotation[0]
    print(project_name)

#process=10 means we used 10 CPU cores (usually CPU will have 8 cores and 16 threads owing to hyperthreading tech)
pool = Pool(processes=10)
argument_list = annotations
result_list_tqdm = []

#by saying iterable=argumnet list, we ask it to loop over the annotations[] list
for result in tqdm(pool.imap_unordered(func=print_images_labels, iterable=argument_list), total=len(argument_list)):
    result_list_tqdm.append(result)
```

In the above code we see that the module 'Pool' is what we use for multi-processing. There are other ways, you can explore later

# Understanding JSON files

A typical json file is shown below. It is just a list of dictionaries

```json
{
  "dataset": 
  [
        {
            "annotations": 
            [
                {
                    "image_path": "images/Project_1098/Y0061712_5.JPG",
                    "original_image_path": "Project_1098/Y0061712.JPG",
                    "bbox_info": 
                    [
                        {
                            "crop_coordinates": [
                                112,
                                1425,
                                856,
                                2169
                            ],
                            "box_coordinates": [
                                315,
                                332,
                                430,
                                413
                            ],
                            "box_attr": {
                                "class": "cotton",
                                "CottonWithDriedFoliage": false,
                                "CottonWithRandomObject": false,
                            },
                            "details": {
                                "completion_date": "",
                                "remarks": "",
                                "project_id": "V2-M-DG-07-23-19-A20-T4-LB-WL-SC-ND-B7",
                            }
                        }
                    ]
                }
            ],
            "dataset_details": 
            {
                "version": "2.0.1",
                "creation_time": "2021-11-22 12:28:41.696893"
            }
        }
  ]
}
```

*Note that if a = [1,2,3,4] and b = {'1':1, '2':2} the method of accessing list and dict elements is a[1] or b['1']* \
**i.e. both a[] and b[] use square brackets**

# Multi-Threading and Finding box IOUs by comparing two images

Using the json above as reference, we can iterate over 'annotations' which is a list of images. Here we will specify the index of the annotations list (to give image)

Also, IOU will be computed for each matching image between rgb and depth. \
The jsons for the same are given below:

![depth_json](/code/orig_sync_first_run_depth.json)
![rgb_json](/code/orig_sync_first_run_rgb.json)

```python
from itertools import repeat
import json
from click import argument
import numpy as np
import copy
import os
import cv2

from multiprocessing import Pool
from tqdm import tqdm

DEV_DEBUG = True

DEPTH_JSON_PATH = '/home/sush/TS/depth_testing/visualisation/29_mar/test/orig_sync_first_run_depth.json'
RGB_JSON_PATH = '/home/sush/TS/depth_testing/visualisation/29_mar/test/orig_sync_first_run_rgb.json'
IOU_DEPTH_JSON_PATH = '/home/sush/TS/depth_testing/visualisation/29_mar/test/depth_with_IOU.json'

DEBUG_PATHS = ['/home/sush/TS/depth_testing/visualisation/29_mar/test/depth_boxes',
                '/home/sush/TS/depth_testing/visualisation/29_mar/test/rgb_boxes',]

IMAGES_PATH = '/media/sush/TS-SW/depth-rgb_annotation/orig_sync_first_run'

temp_depth_matrix = np.array([
    3.89132313e-01, 9.95275561e-03, -1.80530840e+01,
    -1.13047061e-02, 4.02265886e-01, 9.35364939e+01,
    -2.31427742e-05, 1.56034534e-05, 1.00000000e+00
    ])
    
DEPTH_TRANS_MATRIX = np.reshape(temp_depth_matrix,(3,3))

def get_depth_data():
    depth_dict = json.load(open(DEPTH_JSON_PATH,'r'))
    return depth_dict

def transform_rgb_json():
    '''
    This function will open both RGB json and transform the boxes using homography transformation
    '''
    rgb_dict = json.load(open(RGB_JSON_PATH,'r'))
    transformed_dict = copy.deepcopy(rgb_dict)
    images = rgb_dict['annotations']
    for i in range(len(images)):
        for j in range(len(images[i]['bbox_info'])):
            box_coords = images[i]['bbox_info'][j]['box_coordinates']
            x1, y1, x2, y2 = box_coords
            point1 = (x1,y1)
            point2 = (x2,y2)
            x1t, y1t = transform_bbox(point1)
            x2t, y2t = transform_bbox(point2)
            transformed_box_coords = [x1t, y1t, x2t, y2t]
            transformed_dict['annotations'][i]['bbox_info'][j]['box_coordinates'] = transformed_box_coords
    print("finished transforming rbg json")
    return transformed_dict, rgb_dict

def transform_bbox(point):
    """
    Transform a point using the transformation matrix

    Args:
        point (tuple): (x, y) coords of a point
        transform_matrix (np.array): 3x3 transformation matrix
    Returns:
        tuple: (x, y) coords of the transformed point
    """
    # Create [x, y, 0] 3x1 vector for easy dot product in next step
    box_coord = np.array([[point[0]], [point[1]], [1]])

    # Transform the point using dot product. It gives [x', y', s] where s- scaling
    transformed_point = np.dot(DEPTH_TRANS_MATRIX, box_coord)

    # Divide x, y by the scaling factor
    x = int(transformed_point[0] / transformed_point[2])
    y = int(transformed_point[1] / transformed_point[2])
    return x, y

def record_IOU(depth_dict, rgb_dict):
    images = rgb_dict['annotations']
    IOU_tracker = []
    for rgb_i in range(len(images)):
        required_image = images[rgb_i]['original_image_path']
        for dep_i in range(len(depth_dict['annotations'])):
            if depth_dict['annotations'][dep_i]['original_image_path'] == required_image:
                # maybe need deepcopy for below 2 lines?
                rgb_boxes = rgb_dict['annotations'][rgb_i]['bbox_info']
                depth_boxes = depth_dict['annotations'][dep_i]['bbox_info']
                if len(rgb_boxes) == len(depth_boxes):
                    IOU_for_depth_boxes = find_best_IOU(depth_boxes, rgb_boxes)
                    # iterate over depth boxes to add IOU for each box
                    if len(IOU_for_depth_boxes) == len(depth_boxes):
                        for j in range(len(IOU_for_depth_boxes)):
                            depth_dict['annotations'][dep_i]['bbox_info'][j]['box_attr']["IOU_with_RGB"] = IOU_for_depth_boxes[j]
                            IOU_tracker.append(IOU_for_depth_boxes[j])
                    else:
                        for j in range(len(IOU_for_depth_boxes)):
                            depth_dict['annotations'][dep_i]['bbox_info'][j]['box_attr']["IOU_with_RGB"] = "improper_matching"
                            IOU_tracker.append(IOU_for_depth_boxes[j])
    
    return depth_dict, IOU_tracker

def find_best_IOU(depth_boxes, rgb_boxes):
    '''
    Calculate IOU of each box combination

    Args:
        Takes multiple depth boxes (and same number of rgb boxes)
    Returns:
        IOU for each box in form of a list whose len = no. of boxes
    '''
    depth_boxes_IOU = []
    for i in range(len(depth_boxes)):
        current_dbox = depth_boxes[i]["box_coordinates"]
        possible_dbox_IOUs = []
        for j in range(len(rgb_boxes)):
            candidate_rbox = rgb_boxes[j]["box_coordinates"]
            possible_dbox_IOUs.append(calc_iou(current_dbox, candidate_rbox))
        depth_boxes_IOU.append(np.max(np.array(possible_dbox_IOUs)))
    
    print("Box IOU's are: ", depth_boxes_IOU)
    return depth_boxes_IOU

def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return round(iou,3)

def write_meta(new_dict, new_json_path=IOU_DEPTH_JSON_PATH):
        with open(new_json_path, "w") as outfile:
            json.dump(new_dict, outfile)

def calc_average_IOU(IOU_list):
    IOU = np.array(IOU_list)
    IOU= IOU[IOU != 0.0]
    avg_IOU = np.mean(IOU)
    print("average IOU for all boxes is :", avg_IOU)

def collate_debug_images(trans_rgb_dict, depth_dict, orig_rgb_dict):
    # create folders to put debug images in
    for path in DEBUG_PATHS:
        if not os.path.exists(path):
            os.makedirs(path)
    '''
    wrtiing rgb to disk
    '''
    # define threads for multi-threading
    pool = Pool(15)
    # multi-threading for processing rgb images
    rgb_argument_list = range(len(orig_rgb_dict['annotations'])) 
    with pool as p:
        p.starmap(write_rgb_to_disk, zip(rgb_argument_list, repeat(orig_rgb_dict)))
    print("finished writing rgb images to disk")
    
    '''
    writing depth to disk
    '''
    # define threads for multi-threading
    pool = Pool(15)
    # multi-threading for processing depth images
    depth_argument_list = range(len(trans_rgb_dict['annotations']))
    with pool as p:
        p.starmap(write_depth_to_disk, zip(depth_argument_list, repeat(trans_rgb_dict), repeat(depth_dict)))
    print("finished writing depth images to disk")

def write_depth_to_disk(rgb_i, trans_rgb_dict, depth_dict):
    required_image = trans_rgb_dict['annotations'][rgb_i]['original_image_path']
    depth_img = None
    for dep_i in range(len(depth_dict['annotations'])):
        if depth_dict['annotations'][dep_i]['original_image_path'] == required_image:
            # boxes extracted from RGB image annotation
            trans_bboxes = trans_rgb_dict['annotations'][rgb_i]['bbox_info']
            # boxes from depth annotation
            ann_bboxes = depth_dict['annotations'][dep_i]['bbox_info']
            
            depth_img_path = os.path.join(IMAGES_PATH, "depth_pts", "images", required_image)
            depth_img = cv2.imread(depth_img_path, 1)
            
            for j1 in range(len(trans_bboxes)):
                x1,y1,x2,y2 = trans_bboxes[j1]['box_coordinates']
                depth_img = cv2.rectangle(depth_img, (x1,y1), (x2,y2),(30,255,30), 2)
            for j2 in range(len(ann_bboxes)):
                x1,y1,x2,y2 = ann_bboxes[j2]['box_coordinates']
                depth_img = cv2.rectangle(depth_img, (x1,y1), (x2,y2),(255,0,30), 2)

    if depth_img is not None:
        cv2.imwrite((os.path.join(DEBUG_PATHS[0],required_image)),depth_img)

def write_rgb_to_disk(index, orig_rgb_dict):
    rgb_img = None
    rgb_img_name = orig_rgb_dict['annotations'][index]["original_image_path"]
    rgb_img = cv2.imread(os.path.join(IMAGES_PATH, "camtop_raw", "images", 
                                        rgb_img_name), 1)
    bboxes = orig_rgb_dict['annotations'][index]['bbox_info']
    for j in range(len(bboxes)):
        x1,y1,x2,y2 = bboxes[j]['box_coordinates']
        rgb_img = cv2.rectangle(rgb_img, (x1,y1), (x2,y2),(30,255,30), 3)
    
    if rgb_img is not None:
        cv2.imwrite((os.path.join(DEBUG_PATHS[1],rgb_img_name)),rgb_img)
            

if __name__ == '__main__':
    transformed_rgb_dict, orig_rgb_dict = transform_rgb_json()
    depth_dict = get_depth_data()
    depth_dict_with_IOU, IOU_list = record_IOU(depth_dict, transformed_rgb_dict)
    calc_average_IOU(IOU_list)
    write_meta(depth_dict)
    if DEV_DEBUG:
        collate_debug_images(transformed_rgb_dict, depth_dict, orig_rgb_dict)
```

In the above script, the *collate debug images* function uses pool to specify the max no. of threads before each multi-threading run.

This multi-threading takes an iterable and a specific target function just like multi-processing.

Also, since the jsons are input to the target function multiple times (without any change), they can be zipped and iterated using **repeat** as shown:

```python
def collate_debug_images(trans_rgb_dict, depth_dict, orig_rgb_dict):
    # create folders to put debug images in
    for path in DEBUG_PATHS:
        if not os.path.exists(path):
            os.makedirs(path)
    '''
    wrtiing rgb to disk
    '''
    # define threads for multi-threading
    pool = Pool(15)
    # multi-threading for processing rgb images
    rgb_argument_list = range(len(orig_rgb_dict['annotations'])) 
    with pool as p:
        p.starmap(write_rgb_to_disk, zip(rgb_argument_list, repeat(orig_rgb_dict)))
    print("finished writing rgb images to disk")
    
    '''
    writing depth to disk
    '''
    # define threads for multi-threading
    pool = Pool(15)
    # multi-threading for processing depth images
    depth_argument_list = range(len(trans_rgb_dict['annotations']))
    with pool as p:
        p.starmap(write_depth_to_disk, zip(depth_argument_list, repeat(trans_rgb_dict), repeat(depth_dict)))
    print("finished writing depth images to disk")
```


# User Inputs using argparse

This is a simple libraray which allows user to give arguments when running code along with some *tags*

In the below example, we will see some tags such as:
- '-j' is the tag the user has to type in while running the script followed by 1 space and it's required value \
example '-j ./v2.0.2/trial.json'

- '--json_loc' will be the identifier which argparse will return

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_loc', default='./v2.0.1/dev_2.0.1.json', required=False,
                        help='Path to json file')
    parser.add_argument('-ir', '--image_remove', required=False, default=None,
                        nargs='*',help='Image Attributes which should not contribute to creating json')
    parser.add_argument('-br', '--box_remove', required=False, default=None,
                        nargs='*', help='Box attributes which should not contribute to creating json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    json_path = args.json_loc
    #image_remover_list is of type list
    image_remover_list = args.image_remove

```

Also observe that the 'ir' and 'br' tags have an extra argument called **nargs**, which modifies the type of user input

In our case, user input '-ir CottonBlurred CottonShadow' will return a list = ['CottonBlurred', 'CottonShadow']

Furthermore, initialising json_path is like initialising any other variable

# Working with .yaml files

- These files are commonly used to save parameters in a dictionary format
- They can directly convert a python dictionary to a yaml and dump the file in a folder

## Simple script to show working of yaml library in python

```python
## init py
import yaml

## init data
dct_str = {'Pools': [['10', ' 10.127.128.0', ' 18'],
                     ['20', ' 10.127.129.0', ' 19'],
                     ['30', ' 10.127.130.0', ' 20']]}

## print result
print yaml.safe_dump(dct_str)
```

Output:
```
Pools:
- ['10', ' 10.127.128.0', ' 18']
- ['20', ' 10.127.129.0', ' 19']
- ['30', ' 10.127.130.0', ' 20']
```

## Example yaml file

```yaml
# Where to put output files
path_prefix: "/home/workspace/processing/"

###############################################################################
# Overall directories
###############################################################################

stage0_extraction_dir: "stage0_extraction"
stage1_segmentation_dir: "stage1_segmentation"
stage2_depth_dir: "stage2_depth"
stage3_qr_detection_dir: "stage3_qr_detection"
stage4_foreground_dir: "stage4_foreground"
stage5_icp_tightening_dir: "stage5_icp_tightening"
stage6_startergraph_dir: "stage6_startergraph"
stage7_min_span_tree_dir: "stage7_min_span_tree"
stage8_ellipse_likelihoods_dir: "stage8_ellipse_likelihoods"
stage9_ellipse_skeletonization_dir: "stage9_ellipse_skeletonization"
stage10_skmetric_dir: "stage10_skeleton_metrics"
# stage6_pole_dir: "stage6_pole"
# stage7_mesh_skeleton_dir: "stage7_mesh_skeleton"


###############################################################################
# Settings
###############################################################################

stage5_trust_settings:
# High removal
- - search_radius: 0.05
    cutoff_ratio: 1.25
    search_ratio: 1


# Decides whether or not to apply density (neighbor count) to the starter graph
# so that denser areas have lower cost
# THIS IS A LIST and you can run with both if you want (branch)
stage6_use_density:
- true

# This is the size of the voxellization that is used for the fully connected
# starter graph. Finer will obviously result in a longer runtime
stage6_starter_voxel_sizes:
- 0.02

# These are the cloud(s) that will be used to assess the quality of a skeleton.
# I think they should be more full than the clouds used to generate them.
# THIS IS A LIST and you can concatenate multiple clouds together.
stage7_assessment_clouds:
- "class_1_skeldilate_0.ply"
```

