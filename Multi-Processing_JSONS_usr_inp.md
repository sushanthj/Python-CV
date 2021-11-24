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

import module only pulls the module as an object and like any other object, it has only attributes

```python
from multiprocessing import Pool
from tqdm import tqdm
#tqdm is used to show the progress bar, it's just a wrapper around a for loop as seen below

#We define a random list
annotations = [{'project_name':'1'}, {'project_name':'2'}]

def print_images_labels(annotation):
    project_name = annotation[]
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